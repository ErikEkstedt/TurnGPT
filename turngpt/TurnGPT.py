from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# from turngpt.proximity_loss import proximity_loss
from turngpt.models import Attention1D
from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.gpt_mini import GPT
from turngpt.models.proximity import ProxTransformer
from turngpt.models.gpt_rnn import RNN
from turngpt.models.proximity import ProxRNN
from turngpt.acoustic_model import AcousticTransformer, AcousticConv


from transformers import AdamW


def add_lm_model_args(parser):
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == "pretrained":
        from turngpt.models.pretrained import TurnGPTModel

        parser = TurnGPTModel.add_model_specific_args(parser)
    elif temp_args.model == "mini":
        parser = GPT.add_model_specific_args(parser)
    elif temp_args.model == "rnn":

        parser = RNN.add_model_specific_args(parser)
    else:
        raise NotImplementedError(
            'The model argument must be one of "mini", "sparse" or "pretrained"'
        )
    return parser


def add_acoustic_model_args(parser):
    temp_args, _ = parser.parse_known_args()
    if temp_args.acoustic_model == "transformer":
        parser = AcousticTransformer.add_model_specific_args(parser)
    elif temp_args.acoustic_model == "conv":
        parser = AcousticConv.add_model_specific_args(parser)
    elif temp_args.proximity_model is None:
        pass
    else:
        raise NotImplementedError(
            'The acoustic_model argument must be one of "transformer" or "conv"'
        )
    return parser


def add_proximity_model_args(parser):
    temp_args, _ = parser.parse_known_args()
    # Proximity Model
    if temp_args.proximity_model == "transformer":
        parser = ProxTransformer.add_model_specific_args(parser)
    elif temp_args.proximity_model == "rnn":
        parser = ProxRNN.add_model_specific_args(parser)
    elif temp_args.proximity_model is None:
        pass
    else:
        raise NotImplementedError(
            'The proximity_model argument must be one of "transformer" or "rnn"'
        )
    return parser


def build_lm_model(args, n_vocab):
    if args.model == "pretrained":
        from turngpt.models.pretrained import TurnGPTModel

        lm = TurnGPTModel(
            n_vocab=n_vocab,
            dropout=args.dropout,
            pretrained=args.pretrained,
        )
    elif args.model == "mini":

        lm = GPT(
            n_vocab=n_vocab,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            embd_pdrop=args.embd_pdrop,
            resid_pdrop=args.resid_pdrop,
            attn_pdrop=args.attn_pdrop,
            use_speaker_emb=args.use_speaker_emb,
            chunk_size=args.chunk_size,
        )
    elif args.model == "rnn":
        from turngpt.models.gpt_rnn import RNN

        lm = RNN(
            n_vocab=n_vocab,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            dropout=args.dropout,
            rnn=args.rnn,
        )
    else:
        raise NotImplementedError
    return lm


def build_proximity_model(args, lm_model_n_embd):
    if args.proximity_model == "transformer":
        proximity_model = ProxTransformer(
            input_size=lm_model_n_embd,
            hidden_size=args.prox_hidden,
            n_layer=args.prox_layers,
            n_head=args.prox_heads,
            resid_pdrop=args.prox_resid_pdrop,
            attn_pdrop=args.prox_attn_pdrop,
            chunk_size=args.prox_chunk_size,
        )
    elif args.proximity_model == "rnn":
        model = ProxRNN(
            input_size=lm_model_n_embd,
            hidden_size=args.prox_hidden,
            output_size=args.prox_output,
            num_layers=args.prox_layers,
        )
    else:
        proximity_model = None
    return proximity_model


def build_acoustic_model(args):
    if args.acoustic_model == "transformer":
        return AcousticTransformer(
            frames=args.acoustic_frames,
            n_feats=args.acoustic_n_feats,
            enc_hidden=args.acoustic_enc_hidden,
            enc_layers=args.acoustic_enc_layers,
            hidden=args.acoustic_hidden,
            n_head=args.acoustic_n_head,
            n_layer=args.acoustic_n_layer,
            resid_pdrop=args.acoustic_resid_pdrop,
            attn_pdrop=args.acoustic_attn_pdrop,
            chunk_size=args.acoustic_chunk_size,
        )
    elif args.acoustic_model == "conv":
        return AcousticConv(
            frames=args.acoustic_frames,
            n_feats=args.acoustic_n_feats,
            enc_hidden=args.acoustic_enc_hidden,
            enc_layers=args.acoustic_enc_layers,
            hidden=args.acoustic_hidden,
        )
    else:
        return None


def build_modal_mixer(input_size=768, n_modes=2, n_heads=8):
    return Attention1D(
        D=input_size, features=n_modes, features_out=1, num_heads=n_heads
    )


class TurnGPT(pl.LightningModule):
    def __init__(
        self,
        n_vocab=50259,
        pad_idx=None,
        sp1_idx=None,
        sp2_idx=None,
        proximity_horizon=None,
        proximity_constant=1.0,
        learning_rate=1e-4,
        args=None,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.pad_idx = pad_idx
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.proximity_horizon = proximity_horizon
        self.proximity_constant = proximity_constant
        self.learning_rate = learning_rate

        self.lm_model = build_lm_model(args, self.n_vocab)
        self.proximity_model = build_proximity_model(args, self.lm_model.n_embd)
        self.acoustic_model = build_acoustic_model(args)

        self.acoustic_projection = None
        self.modal_mixer = None
        if self.acoustic_model is not None:
            # make sure that the modal mixer gets same input dimension from text and audio
            if self.acoustic_model.hidden != self.lm_model.n_embd:
                self.acoustic_projection = torch.nn.Linear(
                    self.acoustic_model.hidden, self.lm_model.n_embd
                )
            self.modal_mixer = build_modal_mixer(
                input_size=self.lm_model.n_embd, n_modes=2, n_heads=8
            )

        self.save_hyperparameters()

    def forward(
        self,
        input_ids,
        speaker_ids,
        waveform=None,
        spf=None,
        output_attentions=False,
        **kwargs,
    ):
        """ labels are the the same as input_ids shift and padding fix inside model"""

        # LM
        out = self.lm_model(input_ids, speaker_ids)

        # Acoustic
        if spf is not None and self.acoustic_model is not None:
            za, attn_feat = self.acoustic_model(spf)

            if self.acoustic_model.hidden != self.lm_model.n_embd:
                za = self.acoustic_projection(za)

            out["za"] = za
            if output_attentions:
                out["attn_feat"] = attn_feat

            z_mix = torch.stack((out["z"], za), dim=-2)
            z_mix, attn_mix = self.modal_mixer(z_mix)
            z_mix = z_mix.squeeze(-2)
            attn_mix = attn_mix.squeeze(-2)
            out["zm"] = z_mix
            if output_attentions:
                out["attn_mix"] = attn_mix
        else:
            z_mix = out["z"]

        # Proximity
        if self.proximity_model is not None:
            out["proximity_logits"] = self.proximity_model(z_mix)

        return out

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.learning_rate,
            correct_bias=True,
        )

    @torch.no_grad()
    def sample(
        self,
        input_ids,
        speaker_ids,
        batch_size=-1,
        steps=50,
        top_k=5,
        temperature=1.0,
        sample=True,
        stop_at_turn_shift=True,
        max_context=100,
        use_pbar=False,
    ):
        """heavily basged on minGPT/utils in https://github.com/karpathy/minGPT """

        # collect samples and remove from processing when reaching a speaker token
        if stop_at_turn_shift:
            ks = []
            output_samples = []
            output_speaker = []

        # Get more samples by repeating the input by 'batch_size'
        # e.g. (1, 20) -> (10, 20) with batch_size=10
        # or   (5, 20) -> (50, 20) with batch_size=10
        if batch_size > 1:
            input_ids = torch.cat([input_ids] * batch_size)
            speaker_ids = torch.cat([speaker_ids] * batch_size)

        all_input_ids = input_ids.clone()  # store all data even if passt the block size
        all_speaker_ids = None
        if speaker_ids is not None:
            all_speaker_ids = (
                speaker_ids.clone()
            )  # store all data even if passt the block size
        if use_pbar:
            pbar = tqdm(range(steps), desc="TurnGPT Sampling")
        else:
            pbar = range(steps)

        input_ids = input_ids.to(self.device)
        speaker_ids = speaker_ids.to(self.device)

        # past = None
        for k in pbar:
            input_ids = input_ids[:, -max_context:]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[:, -max_context:]

            # TODO: use past in forward pass
            out = self(input_ids, speaker_ids=speaker_ids)
            lm_logits = (
                out["logits"][:, -1, :] / temperature
            )  # Prediction + scale w/ temp

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                lm_logits, idx = lm_logits.topk(k=top_k, dim=-1)
            probs = F.softmax(lm_logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                next_prob_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_idx = idx[
                    torch.arange(len(next_prob_idx)), next_prob_idx
                ].unsqueeze(-1)
                # next_idx = idx[:, next_prob_idx].unsqueeze(-1)
            else:
                next_idx = idx[:, :1]  # top choice

            if speaker_ids is not None:
                # fix speaker ids. Loop (only looking at last token so only looping over batch size -> fastish)
                next_speaker_ids = torch.zeros_like(next_idx)
                for i, (next_id, last_speaker) in enumerate(
                    zip(next_idx, speaker_ids[:, -1])
                ):
                    if next_id == self.sp1_idx or next_id == self.sp2_idx:
                        next_speaker = next_id
                    else:
                        next_speaker = last_speaker
                    next_speaker_ids[i] = last_speaker

            # append to the sequence and continue
            # past = transformer_outputs[1]
            input_ids = torch.cat((input_ids, next_idx), dim=-1)
            all_input_ids = torch.cat((all_input_ids, next_idx.cpu()), dim=-1)

            if speaker_ids is not None:
                speaker_ids = torch.cat((speaker_ids, next_speaker_ids), dim=-1)
                all_speaker_ids = torch.cat(
                    (all_speaker_ids, next_speaker_ids.cpu()), dim=-1
                )

            # print(next_idx)
            # print(all_input_ids.shape)
            if stop_at_turn_shift:
                sp1_batch, _ = torch.where(next_idx == self.sp1_idx)
                sp2_batch, _ = torch.where(next_idx == self.sp2_idx)
                sp_batch = torch.cat((sp1_batch, sp2_batch))
                if len(sp_batch) > 0:
                    # print('pre: ', all_input_ids.shape)
                    for i, b in enumerate(sp_batch):
                        ks.append(k + 1)
                        output_samples.append(all_input_ids[b])
                        output_speaker.append(all_speaker_ids[b])
                    keep_batches = [
                        ind
                        for ind in range(all_input_ids.shape[0])
                        if ind not in sp_batch
                    ]
                    input_ids = input_ids[keep_batches]
                    speaker_ids = speaker_ids[keep_batches]
                    all_input_ids = all_input_ids[keep_batches]
                    all_speaker_ids = all_speaker_ids[keep_batches]
                if len(all_input_ids) == 0:
                    return {
                        "output_ids": output_samples,
                        "output_speaker": output_speaker,
                        "k": ks,
                    }
        return {
            "output_ids": all_input_ids,
            "output_speaker": all_speaker_ids,
            "k": [k + 1] * all_input_ids.shape[0],
        }

    @torch.no_grad()
    def loss_function_turn_shift(self, logits, labels):
        sp_prob = F.softmax(logits, dim=-1)
        sp_prob = torch.stack(
            (sp_prob[..., self.sp1_idx], sp_prob[..., self.sp2_idx]), dim=-1
        )
        sp_prob = sp_prob.max(dim=-1).values
        sp_prob = torch.stack((sp_prob, 1 - sp_prob), dim=-1)
        sp_prob = sp_prob.view(-1, sp_prob.size(-1))

        sp_labels = (labels == self.sp1_idx).float()
        sp_labels[labels == self.sp2_idx] = 1
        sp_labels = torch.stack((sp_labels, 1 - sp_labels), dim=-1)
        sp_labels = sp_labels.view(-1, sp_labels.size(-1))

        if self.pad_idx is not None:
            sp_labels = sp_labels[labels.view(-1) != self.pad_idx]
            sp_prob = sp_prob[labels.view(-1) != self.pad_idx]
        # loss = F.binary_cross_entropy(sp_prob, sp_labels)  # does not work with half precision used elsewhere?
        loss = torch.nn.BCEWithLogitsLoss()(sp_prob, sp_labels)
        return loss

    def loss_function_lm(self, logits, labels):
        if self.pad_idx is not None:
            labels[labels == self.pad_idx] = -100  # don't train on these
        return torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

    def shared_step(self, batch):
        """
        Simply shifts the input to acquire the labels.
        Sets waveform/spf:speech-features to None if they don't exist
        """
        input_ids, speaker_ids = batch[0], batch[1]

        # shift for labels
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        ret = {
            "input_ids": input_ids,
            "speaker_ids": speaker_ids,
            "labels": labels,
        }

        if len(batch) > 2:
            ret["waveform"] = batch[2][:, :-1].contiguous()

        if len(batch) > 3 and len(batch[3]) > 0:
            ret["spf"] = batch[3][:, :-1].contiguous()

        if len(batch) > 4 and len(batch[4]) > 0:
            ret["post_silence"] = batch[4][:, :-1].contiguous()

        return ret

    def training_step(self, batch, *args, **kwargs):
        b = self.shared_step(batch)

        # LM
        out = self(b["input_ids"], b["speaker_ids"], b["waveform"], b["spf"])
        loss = self.loss_function_lm(out["logits"], b["labels"])

        if "proximity_logits" in out:
            prox_losses = self.proximity_model.loss_function(
                out["proximity_logits"],
                b["labels"],
                self.sp1_idx,
                self.sp2_idx,
                self.pad_idx,
            )
            self.log(
                "avg_train_prox_loss",
                prox_losses["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "train_prox_loss",
                prox_losses["loss"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "avg_train_lm_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            # add LM loss and proximity loss
            loss += self.proximity_constant * prox_losses["loss"]

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "avg_train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        """for multi-gpu"""
        return {"loss": batch_parts["loss"].mean()}

    def validation_step(self, batch, *args, **kwargs):
        b = self.shared_step(batch)

        # LM
        out = self(b["input_ids"], b["speaker_ids"], b["waveform"], b["spf"])
        loss = self.loss_function_lm(out["logits"], b["labels"])

        sp_loss = self.loss_function_turn_shift(out["logits"], b["labels"])

        self.log("val_lm_loss", loss, logger=True)
        self.log("val_sp_loss", sp_loss)

        if "proximity_logits" in out:
            prox_losses = self.proximity_model.loss_function(
                out["proximity_logits"],
                b["labels"],
                self.sp1_idx,
                self.sp2_idx,
                self.pad_idx,
            )
            self.log("val_prox_loss", prox_losses["loss"], logger=True)

            # add LM loss and proximity loss
            loss += self.proximity_constant * prox_losses["loss"]

        self.log("val_loss", loss, logger=True)
        return {"val_loss": loss}

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    @staticmethod
    def add_model_specific_args(
        parent_parser, lm_model="pretrained", proximity_model=None, acoustic_model=None
    ):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default=lm_model,  # sparse, pretrained
        )
        parser.add_argument(
            "--acoustic_model",
            type=str,
            default=acoustic_model,  # sparse, pretrained
        )
        parser.add_argument(
            "--proximity_model",
            type=str,
            default=proximity_model,  # sparse, pretrained
        )
        parser.add_argument(
            "--proximity_constant",
            type=float,
            default=1.0,
        )

        # Training
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--early_stopping", action="store_true")
        parser.add_argument("--patience", default=10, type=int)

        # Language Model
        parser = add_lm_model_args(parser)

        # Acoustic Model
        parser = add_acoustic_model_args(parser)

        # Proximity Model
        parser = add_proximity_model_args(parser)
        return parser


if __name__ == "__main__":
    from os.path import join
    from os import environ
    from turngpt.acousticDM import AcousticGPTDM
    from ttd.utils import get_run_dir

    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(
        parser, proximity_model="transformer", acoustic_model="transformer"
    )
    tmp_args, _ = parser.parse_known_args()
    print("\nModel Settings")
    print("--------------")
    for k, v in vars(tmp_args).items():
        print(f"{k}: {v}")
    parser = AcousticGPTDM.add_data_specific_args(parser, datasets=["maptask"])
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.prosody = True
    args.acoustic_hidden = 128

    dm = AcousticGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    loader = dm.val_dataloader()
    batch = next(iter(loader))
    input_ids, speaker_ids, xa, xp = batch
    print("input_ids: ", tuple(input_ids.shape))
    print("speaker_ids: ", tuple(speaker_ids.shape))
    print("xa: ", tuple(xa.shape))
    print("xp: ", tuple(xp.shape))

    model = TurnGPT(
        n_vocab=50259,
        pad_idx=50256,
        sp1_idx=50257,
        sp2_idx=50258,
        learning_rate=args.learning_rate,
        proximity_constant=args.proximity_constant,
        args=args,
    )

    print("LM: ", model.lm_model.__class__.__name__)
    if model.acoustic_model is not None:
        print("Acoustic: ", model.acoustic_model.__class__.__name__)
    if model.acoustic_projection is not None:
        print("Acu-proj: ", model.acoustic_projection.__class__.__name__)
    if model.proximity_model is not None:
        print("Proximity: ", model.proximity_model.__class__.__name__)

    # Where to save the training
    print()
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print(args.save_dir)
    print()

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)

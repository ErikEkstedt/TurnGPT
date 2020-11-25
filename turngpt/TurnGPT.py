from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# from turngpt.proximity_loss import proximity_loss
from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.gpt_mini import GPT
from turngpt.models.proximity import ProxTransformer
from turngpt.models.proximity import ProxRNN

# from acousticgpt.acoustic_gpt import AcousticGPT

from transformers import AdamW


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


def build_acoustic_model(args, lm_n_embd):
    if args.acoustic_model == "transformer":
        from acousticgpt.acoustic_gpt import AcousticModel

        return AcousticModel(
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
    else:
        return None


def build_modal_mixer(input_size=768, n_modes=2, n_heads=8):
    from acousticgpt.acoustic_gpt import Attention1D

    return Attention1D(
        D=input_size, features=n_modes, features_out=1, num_heads=n_heads
    )


# TODO: better processs_bar logging not really that important
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

        if args.acoustic_model is not None:
            self.acoustic_model = build_acoustic_model(args, self.lm_model.n_embd)
            if self.acoustic_model.hidden != self.lm_model.n_embd:
                self.upsample = torch.nn.Linear(
                    self.acoustic_model.hidden, self.lm_model.n_embd
                )
            self.modal_mixer = build_modal_mixer(
                input_size=self.lm_model.n_embd, n_modes=2, n_heads=8
            )

        self.save_hyperparameters()

    def freeze_lm_model(self):
        self.lm_model.freeze()
        print("LM Frozen")

    def freeze_proximity_model(self):
        self.proximity_model.freeze()
        print("Proximity Frozen")

    def freeze_acoustic_model(self):
        self.acoustic_model.freeze()
        print("Acoustic Frozen")

    def freeze_modal_mixer(self):
        for p in self.modal_mizer.parameters():
            p.requires_grad = False
        print("Modal Mixer Frozen")

    def forward(self, input_ids, speaker_ids, **kwargs):
        """ labels are the the same as input_ids shift and padding fix inside model"""
        out = self.lm_model(input_ids, speaker_ids)
        if self.proximity_model is not None:
            out["proximity_logits"] = self.proximity_model(out["z"])
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
        sp1_idx=None,
        sp2_idx=None,
        steps=50,
        top_k=5,
        temperature=1.0,
        sample=True,
        stop_at_turn_shift=True,
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
            input_ids = input_ids[:, -self.block_size :]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[:, -self.block_size :]

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
                    if next_id == sp1_idx or next_id == sp2_idx:
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
                sp1_batch, _ = torch.where(next_idx == sp1_idx)
                sp2_batch, _ = torch.where(next_idx == sp2_idx)
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

    def loss_function(self, logits, labels):
        if self.pad_idx is not None:
            labels[labels == self.pad_idx] = -100  # don't train on these
        return torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )

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

    def training_step(self, batch, *args, **kwargs):
        input_ids, speaker_ids = batch[0], batch[1]

        # Forward pass
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()

        audio, prosody = None, None
        if len(batch) > 2:
            audio = batch[2][:, 1:].contiguous()

        if len(batch) > 3 and len(batch[3]) > 0:
            prosody = batch[3][:, 1:].contiguous()
        else:
            prosody = None

        # LM
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)

        # Audio
        if prosody is not None and self.acoustic_model is not None:
            za = self.acoustic_model(prosody)
            if self.acoustic_model.hidden != self.lm_model.n_embd:
                za = self.upsample(za)
            z_mix = torch.stack((output["z"], za), dim=-2)
            z_mix, attn_mix = self.modal_mixer(z_mix)
            z_mix = z_mix.squeeze(-2)
            attn_mix = attn_mix.squeeze(-2)
        else:
            z_mix = output["z"]

        if self.proximity_model is not None:
            prox_logits = self.proximity_model(z_mix)
            prox_losses = self.proximity_model.loss_function(
                prox_logits, labels, self.sp1_idx, self.sp2_idx, self.pad_idx
            )
            prox_loss = 0
            for l in prox_losses:
                prox_loss += l

            self.log(
                "avg_train_prox_loss",
                prox_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
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

            # multiply by constant
            loss += self.proximity_constant * prox_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
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
        input_ids, speaker_ids = batch[0], batch[1]

        # Forward pass
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()

        audio, prosody = None, None
        if len(batch) > 2:
            audio = batch[2][:, 1:].contiguous()

        if len(batch) > 3 and len(batch[3]) > 0:
            prosody = batch[3][:, 1:].contiguous()
        else:
            prosody = None

        # LM
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)

        # Audio
        if prosody is not None and self.acoustic_model is not None:
            za = self.acoustic_model(prosody)
            if self.acoustic_model.hidden != self.lm_model.n_embd:
                za = self.upsample(za)
            z_mix = torch.stack((output["z"], za), dim=-2)
            z_mix, attn_mix = self.modal_mixer(z_mix)
            z_mix = z_mix.squeeze(-2)
            attn_mix = attn_mix.squeeze(-2)
        else:
            z_mix = output["z"]

        if self.proximity_model is not None:
            prox_logits = self.proximity_model(z_mix)
            prox_losses = self.proximity_model.loss_function(
                prox_logits, labels, self.sp1_idx, self.sp2_idx, self.pad_idx
            )
            prox_loss = 0
            for l in prox_losses:
                prox_loss += l

            self.log(
                "avg_val_prox_loss",
                prox_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "avg_val_lm_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            # multiply by constant
            loss += self.proximity_constant * prox_loss

        self.log(
            "val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "avg_val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
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
        parser.add_argument("--early_stopping", default=False, type=bool)
        parser.add_argument("--patience", default=10, type=int)

        temp_args, _ = parser.parse_known_args()

        # Language Model
        if temp_args.model == "pretrained":
            from turngpt.models.pretrained import TurnGPTModel

            parser = TurnGPTModel.add_model_specific_args(parser)
        elif temp_args.model == "mini":
            from turngpt.models.gpt_mini import GPT

            parser = GPT.add_model_specific_args(parser)
        elif temp_args.model == "rnn":
            from turngpt.models.gpt_rnn import RNN

            parser = RNN.add_model_specific_args(parser)
        else:
            raise NotImplementedError(
                'The model argument must be one of "mini", "sparse" or "pretrained"'
            )

        # Acoustic Model
        if temp_args.acoustic_model == "transformer":
            from acousticgpt.acoustic_gpt import AcousticModel

            parser = AcousticModel.add_model_specific_args(parser)

        # Proximity Model
        if temp_args.proximity_model == "transformer":
            from turngpt.models.proximity import ProxTransformer

            parser = ProxTransformer.add_model_specific_args(parser)
        elif temp_args.proximity_model == "rnn":
            from turngpt.models.proximity import ProxRNN

            parser = ProxRNN.add_model_specific_args(parser)
        elif temp_args.proximity_model is None:
            pass
        else:
            raise NotImplementedError(
                'The proximity_model argument must be one of "transformer" or "rnn"'
            )

        return parser


if __name__ == "__main__":

    # parser = ArgumentParser()
    # parser = TurnGPT.add_model_specific_args(parser)
    # args = parser.parse_args()

    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")

    # model = TurnGPT(
    #     n_vocab=50259,
    #     pad_idx=50256,
    #     sp1_idx=50257,
    #     sp2_idx=50258,
    #     learning_rate=args.learning_rate,
    #     proximity_constant=args.proximity_constant,
    #     args=args,
    # )

    # print(model)

    from os.path import join
    from os import environ
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from acousticgpt.acousticDM import AcousticGPTDM
    from ttd.utils import get_run_dir

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AcousticGPTDM.add_data_specific_args(parser, datasets=["maptask"])
    parser = TurnGPT.add_model_specific_args(
        parser, proximity_model="transformer", acoustic_model="transformer"
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.prosody = True

    args.acoustic_hidden = 128

    dm = AcousticGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    # loader = dm.val_dataloader()
    # batch = next(iter(loader))
    # input_ids, speaker_ids, xa, xp = batch
    # print("input_ids: ", tuple(input_ids.shape))
    # print("speaker_ids: ", tuple(speaker_ids.shape))
    # print("xa: ", tuple(xa.shape))
    # print("xp: ", tuple(xp.shape))

    model = TurnGPT(
        n_vocab=50259,
        pad_idx=50256,
        sp1_idx=50257,
        sp2_idx=50258,
        learning_rate=args.learning_rate,
        proximity_constant=args.proximity_constant,
        args=args,
    )
    # print(model)
    # model.cuda()
    # out = model.training_step(batch)

    # Where to save the training
    print()
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print(args.save_dir)
    print()

    checkpoint_callback = None
    callbacks = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        name = "TurnGPT" + args.model
        desc = f"{name} training"
        logger = TensorBoardLogger(args.save_dir, name=name)
        ch_path = join(logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}-{val_loss:.5f}",
            save_top_k=2,
            mode="min",
            monitor="val_loss",
        )

        # Save the used tokenizer
        tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        torch.save(dm.tokenizer, tokenizer_path)
        print("tokenizer saved -> ", tokenizer_path)

        if args.early_stopping:
            print(f"Early stopping (patience={args.patience})")
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=True,
            )
            callbacks = [early_stop_callback]
        print("-" * 50)

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)

    # out = model.validation_step([b.to(model.device) for b in batch])

    # print("z: ", tuple(z.shape))

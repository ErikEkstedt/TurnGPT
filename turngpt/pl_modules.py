from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from turngpt.proximity_loss import proximity_loss


class TurnGPTOld(pl.LightningModule):
    def forward(self, input_ids, speaker_ids, **kwargs):
        """ labels are the the same as input_ids shift and padding fix inside model"""
        return self.model(input_ids, speaker_ids=speaker_ids, **kwargs)

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
            input_ids = input_ids[:, -self.model.block_size :]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[:, -self.model.block_size :]

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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

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
        loss = F.binary_cross_entropy(sp_prob, sp_labels)
        return loss

    def training_step(self, batch, *args, **kwargs):
        input_ids, speaker_ids = batch[0], batch[1]

        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)
        sp_loss = self.loss_function_turn_shift(output["logits"], labels)

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
        self.log(
            "avg_train_sp_loss",
            sp_loss,
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
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)
        sp_loss = self.loss_function_turn_shift(output["logits"], labels)
        self.log("val_loss", loss)
        self.log("val_sp_loss", sp_loss)
        return {"val_loss": loss, "val_sp_loss": sp_loss}


class TurnGPT(pl.LightningModule):
    def __init__(
        self,
        LM,
        proximity_head=None,
        pad_idx=None,
        sp1_idx=None,
        sp2_idx=None,
        learning_rate=1e-4,
        model="pretrained",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx

        self.model = LM
        self.proximity_head = proximity_head

        self.save_hyperparameters(
            "pad_idx", "sp1_idx", "sp2_idx", "learning_rate", "model"
        )

    def forward(self, input_ids, speaker_ids, **kwargs):
        """ labels are the the same as input_ids shift and padding fix inside model"""
        out = self.model(input_ids, speaker_ids)
        if self.proximity_head is not None:
            out["proximity_logits"] = self.proximity_head(out["z"])
        return out

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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

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
        loss = F.binary_cross_entropy(sp_prob, sp_labels)
        return loss

    def training_step(self, batch, *args, **kwargs):
        input_ids, speaker_ids = batch[0], batch[1]

        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)
        sp_loss = self.loss_function_turn_shift(output["logits"], labels)

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
        self.log(
            "avg_train_sp_loss",
            sp_loss,
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
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)
        sp_loss = self.loss_function_turn_shift(output["logits"], labels)
        self.log("val_loss", loss)
        self.log("val_sp_loss", sp_loss)
        return {"val_loss": loss, "val_sp_loss": sp_loss}

    def configure_optimizers(self):
        if self.hparams["model"] == "pretrained":
            from transformers import AdamW

            return AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                correct_bias=True,
            )
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default="pretrained",  # sparse, pretrained
        )

        # Training
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--early_stopping", default=False, type=bool)
        parser.add_argument("--patience", default=10, type=int)
        return parser


if __name__ == "__main__":
    # from matplotlib import use as mpl_use
    # mpl_use("Agg")
    # pl.seed_everything(1234)

    from ttd.basebuilder import add_builder_specific_args
    from turngpt.turngpt_dm import TurnGPTDM

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser = TurnGPT.add_model_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["coached"],
    )

    temp_args, _ = parser.parse_known_args()
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders

    if temp_args.model == "pretrained":
        from turngpt.models.pretrained import TurnGPTModel

        print("pretrained")
        parser = TurnGPTModel.add_model_specific_args(parser)
    else:
        raise NotImplementedError(
            'The model argument must be one of "mini", "sparse" or "pretrained"'
        )

    args = parser.parse_args()
    args.chunk_size = 128
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")

    # ------------------------------------------------------------------
    # Data
    dm = TurnGPTDM(args)
    print("\nDataLoader")
    print("----------")
    print("Batch size:  ", args.batch_size)
    print("num workers: ", args.num_workers)
    print("chunks:      ", args.chunk_size)

    # ------------------------------------------------------------------
    # Trainer
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     logger=logger,
    #     checkpoint_callback=checkpoint_callback,
    #     callbacks=callbacks,
    # )

    # ------------------------------------------------------------------
    # Data
    dm.prepare_data()
    dm.setup("fit")

    # ------------------------------------------------------------------
    # Model
    if args.model == "pretrained":
        from turngpt.models.pretrained import TurnGPTModel

        lm = TurnGPTModel(
            n_vocab=len(dm.tokenizer),
            dropout=args.dropout,
            pretrained=args.pretrained,
        )
    if args.model == "mini":
        from turngpt.models.mini import TurnGPTModel
    else:
        raise NotImplementedError

    model = TurnGPT(
        lm,
        pad_idx=dm.tokenizer.pad_token_id,
        sp1_idx=dm.sp1_idx,
        sp2_idx=dm.sp2_idx,
        learning_rate=args.learning_rate,
        model=args.model,
    )

    trainer = pl.Trainer(fast_dev_run=1, gpus=1)
    trainer.fit(model, datamodule=dm)

    print("\n-----", "Model", "-----")
    print("pad_idx: ", model.pad_idx)
    print("n_vocab: ", len(dm.tokenizer))
    if "n_layer" in args:
        print("n_layer: ", args.n_layer)
    else:
        print("n_layer: ", model.n_layer)

    if "n_head" in args:
        print("n_head: ", args.n_head)
    else:
        print("n_head: ", model.n_head)
    if "n_embd" in args:
        print("n_embd: ", args.n_embd)
    else:
        print("n_embd: ", model.n_embd)
    print()

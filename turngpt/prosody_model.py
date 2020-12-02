from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio.transforms as AT

from turngpt.acoustic_model import SPFConv
from turngpt.models.gpt_mini import Block, TransformerConfig
from turngpt.models.proximity import ProxTransformer

from ttd.tokenizer_helpers import convert_ids_to_tokens
from turngpt.turngpt_utils import get_speaker_shift_indices
from turngpt.models import Attention1D


class SPF(pl.LightningModule):
    def __init__(
        self,
        frames=100,
        n_feats=1,
        enc_hidden=32,
        enc_layers=3,
        hidden=256,
        n_head=8,
        prox_hidden=256,
        prox_horizon=[1, 3],
        prox_n_head=4,
        prox_n_layer=1,
        prox_resid_pdrop=0.1,
        prox_attn_pdrop=0.1,
        prox_chunk_size=128,
        prox_loss_constants=[1, 1],
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        lr=1e-4,
        weight_decay=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.lr = lr
        self.weight_decay = weight_decay

        self.hidden = hidden
        self.n_head = n_head
        self.n_feats = n_feats
        self.feature_encoder = SPFConv(
            frames,
            n_feats=n_feats,
            hidden=enc_hidden,
            output_size=hidden,
            num_layers=enc_layers,
        )

        if n_feats > 1:
            self.channel_transformer = Attention1D(
                D=hidden, features=n_feats, features_out=1, num_heads=n_head
            )

        self.prox = ProxTransformer(
            input_size=hidden,
            hidden_size=prox_hidden,
            horizon=prox_horizon,
            n_head=prox_n_head,
            n_layer=prox_n_layer,
            resid_pdrop=prox_resid_pdrop,
            attn_pdrop=prox_attn_pdrop,
            chunk_size=prox_chunk_size,
            loss_constants=prox_loss_constants,
        )

    def forward(self, x, output_attention=False):
        ret = {}
        x = self.feature_encoder(x)
        if self.n_feats > 1:
            x, feat_attn = self.channel_transformer(x)
            x = x.squeeze(-2)
            if output_attention:
                print(feat_attn.shape)
                ret["feat_attn"] = feat_attn.squeeze(-2).detach().cpu()
        else:
            x = x.squeeze(-2)

        out = self.prox(x, output_attention)
        for k, v in out.items():
            ret[k] = v
        return ret

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def training_step(self, batch, *args, **kwargs):
        batch = self.shared_step(batch)
        out = self(batch["x"])
        prox_losses = self.prox.loss_function(
            out["logits"], batch["labels"], self.sp1_idx, self.sp2_idx, self.pad_idx
        )
        self.log("train_prox_loss", prox_losses["loss"], logger=True)
        return prox_losses["loss"]

    def validation_step(self, batch, *args, **kwargs):
        batch = self.shared_step(batch)
        out = self(batch["x"])
        prox_losses = self.prox.loss_function(
            out["logits"], batch["labels"], self.sp1_idx, self.sp2_idx, self.pad_idx
        )
        self.log(
            "val_prox_loss",
            prox_losses["loss"],
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": prox_losses["loss"]}

    def shared_step(self, batch):
        """
        Simply shifts the input to acquire the labels.
        Sets waveform/spf:speech-features to None if they don't exist
        """

        # shift for labels
        batch["labels"] = labels = batch["input_ids"][:, 1:].contiguous()
        batch["input_ids"] = batch["input_ids"][:, :-1].contiguous()
        batch["speaker_ids"] = batch["speaker_ids"][:, :-1].contiguous()

        if "waveform" in batch:
            batch["waveform"] = batch["waveform"][:, :-1].contiguous()

        if "f0" in batch:
            batch["f0"] = batch["f0"][:, :-1].contiguous()

        if "rms" in batch:
            batch["rms"] = batch["rms"][:, :-1].contiguous()

        if self.n_feats == 2:
            batch["x"] = torch.stack((batch["f0"], batch["rms"]), dim=-1)
        else:
            batch["x"] = batch["f0"].unsqueeze(-1)
        return batch

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--acoustic_frames", type=int, default=100)
        parser.add_argument("--acoustic_n_feats", type=int, default=1)
        parser.add_argument("--acoustic_enc_hidden", type=int, default=32)
        parser.add_argument("--acoustic_enc_layers", type=int, default=3)
        parser.add_argument("--acoustic_hidden", type=int, default=128)
        parser.add_argument("--acoustic_n_head", type=int, default=8)
        parser.add_argument("--acoustic_n_layer", type=int, default=1)
        parser.add_argument("--acoustic_resid_pdrop", type=float, default=0.1)
        parser.add_argument("--acoustic_attn_pdrop", type=float, default=0.1)
        parser.add_argument("--acoustic_chunk_size", type=int, default=128)

        # Training
        parser.add_argument("--learning_rate", type=int, default=1e-4)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser = ProxTransformer.add_model_specific_args(parser)
        return parser


def plot_prox(
    prox_logits, input_ids, tokenizer, sp1_idx, sp2_idx, attn_feat=None, plot=False
):
    inp_tokens = convert_ids_to_tokens(input_ids, tokenizer)[0]
    _, sp_idx = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)

    probs = []
    for head in range(prox_logits.shape[-1]):
        probs.append(torch.sigmoid(prox_logits[..., head]).cpu())

    if attn_feat is not None:
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        b = ax[0]
        a = ax[-1]
        b.plot(attn_feat[:, 0], label="f0")
        b.vlines(sp_idx, ymin=0, ymax=1, color="k", linewidth=2, alpha=0.3)
        b.legend()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), sharex=True)
        a = ax
    # prox
    for i, p in enumerate(probs):
        a.plot(p, label=f"prox {i}", alpha=0.6)
    a.vlines(sp_idx, ymin=0, ymax=1, color="k", linewidth=2, alpha=0.3)
    a.set_xticks(torch.arange(len(inp_tokens)))
    a.set_xticklabels(inp_tokens, rotation=65, fontsize=10)
    a.set_xlim([0, len(inp_tokens)])

    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


def main():
    from turngpt.acousticDM import AudioDM
    from ttd.utils import get_run_dir
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os import environ
    from os.path import join

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SPF.add_model_specific_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        f0=True,
        rms=True,
        waveform=False,
        normalize_f0=False,
        interpolate_f0=False,
    )
    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    print("normalize f0: ", dm.normalize_f0)
    print("interpolate f0: ", dm.interpolate_f0)
    print("log rms: ", dm.log_rms)
    loader = dm.val_dataloader()

    if args.checkpoint is not None:
        model = SPF.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams)
    else:
        model = SPF(
            frames=args.acoustic_frames,
            n_feats=args.acoustic_n_feats,
            enc_hidden=args.acoustic_enc_hidden,
            enc_layers=args.acoustic_enc_layers,
            hidden=args.acoustic_hidden,
            n_head=args.acoustic_n_head,
            prox_hidden=args.prox_hidden,
            prox_horizon=args.prox_horizon,
            prox_n_head=args.prox_heads,
            prox_n_layer=args.prox_layers,
            prox_resid_pdrop=args.prox_resid_pdrop,
            prox_attn_pdrop=args.prox_attn_pdrop,
            prox_chunk_size=args.prox_chunk_size,
            prox_loss_constants=args.prox_horizon_constants,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            pad_idx=dm.pad_idx,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # ------------------------------------------------------------------
    # Checkpoint callback (early stopping)

    # Where to save the training
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print("root dir: ", args.save_dir)
    checkpoint_callback = None
    callbacks = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        name = "F0Model"
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

    # ------------------------------------------------------------------
    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Fit
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    main()
    import sys

    sys.exit(0)

    # debug
    import matplotlib.pyplot as plt

    from turngpt.acousticDM import AudioDM
    from ttd.utils import get_run_dir
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os import environ
    from os.path import join

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SPF.add_model_specific_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        f0=True,
        rms=True,
        waveform=True,
        normalize_f0=True,
        interpolate_f0=True,
    )
    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()
    batch = next(iter(loader))

    args.checkpoint = "TurnGPT/turngpt/runs/F0Model/version_1/checkpoints/epoch=15-val_loss=0.30830.ckpt"
    args.hparams = "TurnGPT/turngpt/runs/F0Model/version_1/hparams.yaml"

    args.checkpoint = "TurnGPT/turngpt/runs/F0Model/f0ni_rms/checkpoints/epoch=39-val_loss=0.27173.ckpt"
    args.hparams = "TurnGPT/turngpt/runs/F0Model/f0ni_rms/hparams.yaml"

    # args.acoustic_n_feats = 1

    if args.checkpoint is not None:
        model = SPF.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams)
    else:
        model = SPF(
            frames=args.acoustic_frames,
            n_feats=args.acoustic_n_feats,
            enc_hidden=args.acoustic_enc_hidden,
            enc_layers=args.acoustic_enc_layers,
            hidden=args.acoustic_hidden,
            n_head=args.acoustic_n_head,
            prox_hidden=args.prox_hidden,
            prox_horizon=args.prox_horizon,
            prox_n_head=args.prox_heads,
            prox_n_layer=args.prox_layers,
            prox_resid_pdrop=args.prox_resid_pdrop,
            prox_attn_pdrop=args.prox_attn_pdrop,
            prox_chunk_size=args.prox_chunk_size,
            prox_loss_constants=args.prox_horizon_constants,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            pad_idx=dm.pad_idx,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # ------------------------------------------------------------------
    # Checkpoint callback (early stopping)

    batch = next(iter(dm.val_dataloader()))

    b = model.shared_step(batch)

    with torch.no_grad():
        out = model(b["x"], output_attention=True)
    input_ids = batch["input_ids"]
    prox_logits = out["logits"]
    if "feat_attn" in out:
        attn_feat = out["feat_attn"]
        print("attn_feat: ", attn_feat.shape)
    if "prox_attn" in out:
        print("attn_prox: ", out["prox_attn"].shape)

    from ttd.tokenizer_helpers import convert_ids_to_tokens

    tokens = convert_ids_to_tokens(input_ids, dm.tokenizer)

    bi = 0
    N = 12

    print(out["prox_attn"][bi, :, N, : N + 2])

    inp_tokens = tokens[bi, : N + 2]
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # for head in range(8):
    #     ax.plot(out['prox_attn'][bi, head, N, :N+2])
    ax.plot(out["prox_attn"][bi, :, N, : N + 2].sum(dim=0))
    ax.vlines(N, 0, 2, color="r")
    ax.set_xticks(torch.arange(len(inp_tokens)))
    ax.set_xticklabels(inp_tokens, rotation=65, fontsize=10)
    ax.set_xlim([0, len(inp_tokens)])
    plt.pause(0.01)

    for b in range(input_ids.shape[0]):
        fig, ax = plot_prox(
            prox_logits=prox_logits[b],
            input_ids=input_ids[b].unsqueeze(0),
            tokenizer=dm.tokenizer,
            attn_feat=attn_feat[b],
            sp1_idx=model.sp1_idx,
            sp2_idx=model.sp2_idx,
            plot=True,
        )
        input()

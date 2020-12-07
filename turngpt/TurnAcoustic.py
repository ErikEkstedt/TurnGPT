from argparse import ArgumentParser
from os.path import split, join
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ttd.tokenizer_helpers import convert_ids_to_tokens
from turngpt.models import Attention1D
from turngpt.models.prosody import ProsodyEncoder
from turngpt.models.proximity import ProxTransformer, ProxRNN
from turngpt.models.pretrained import TurnGPTModel
from turngpt.turngpt_utils import (
    get_speaker_shift_indices,
    get_positive_and_negative_indices,
)


def hparams_from_checkpoint(checkpoint):
    return join(split(split(checkpoint)[0])[0], "hparams.yaml")


def get_model(args, lm_n_vocab=None):
    if args.checkpoint is None:
        model = TurnAcoustic(
            prosody_encoder=args.prosody_encoder,
            LM=args.LM,
            prosody_frames=args.prosody_frames,
            prosody_n_feats=args.prosody_n_feats,
            prosody_hidden=args.prosody_hidden,
            prosody_layers=args.prosody_layers,
            prosody_kernel=args.prosody_kernel,
            prosody_first_stride=args.prosody_first_stride,
            prosody_n_head=args.prosody_n_head,
            prox_input_size=args.prox_input_size,
            prox_hidden=args.prox_hidden,
            prox_layers=args.prox_layers,
            prox_heads=args.prox_heads,
            prox_horizon=args.prox_horizon,
            prox_horizon_constants=args.prox_horizon_constants,
            prox_resid_pdrop=args.prox_resid_pdrop,
            prox_attn_pdrop=args.prox_attn_pdrop,
            prox_dropout=args.prox_dropout,
            prox_chunk_size=args.prox_chunk_size,
            prox_model=args.proximity_model,
            lm_n_vocab=lm_n_vocab,
            lm_dropout=args.lm_dropout,
            lm_pretrained=args.lm_pretrained,
            lm_patience=args.lm_patience,
            modal_n_heads=args.modal_n_heads,
            log_rms=args.log_rms,
            f0_norm=args.normalize_f0,
            f0_interpolate=args.interpolate_f0,
        )
        if args.lm_checkpoint is not None:
            state_dict = torch.load(args.lm_checkpoint)["state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            print("Loaded LM weights: ", args.lm_checkpoint)
            if len(unexpected_keys) > 0:
                print("UNEXPECTED KEYS")
                for uk in unexpected_keys:
                    print(uk)
                print()
                assert False, "Assuming load did not work..."
    else:
        # loading entire model from args.checkpoing (and hparams.yaml)
        print("Load checkpoint")
        model = TurnAcoustic.load_from_checkpoint(
            args.checkpoint, hparams=hparams_from_checkpoint(args.checkpoint)
        )
    if torch.cuda.is_available():
        model.to("cuda")
    return model


class Plots(object):
    @staticmethod
    def plot_turn_acoustic(
        target_ind,
        trp,
        attn_prox,
        attn_pros,
        attn_mix,
        tokens,
        trp_lm=None,
        fig=None,
        ax=None,
        plot=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex=True)

        x = torch.arange(len(tokens))
        ax[0].plot(x, attn_pros[:, 0], "--", alpha=0.5)
        ax[0].plot(x, attn_pros[:, 1], "--", alpha=0.5)
        ax[0].plot(x, attn_pros[:, 0], ".", alpha=0.7, label="f0")
        ax[0].plot(x, attn_pros[:, 1], ".", alpha=0.7, label="rms")
        ax[0].set_ylim([0, 1])
        ax[0].legend()

        apx = attn_prox[:, :, target_ind, : target_ind + 1].sum(dim=0).sum(dim=0)
        ax[1].bar(torch.arange(apx.shape[0]), apx, label="attention time")
        ax[1].set_ylim([0, 1])
        ax[1].legend()

        ax[2].plot(x, attn_mix[:, 0], "--", alpha=0.5)
        ax[2].plot(x, attn_mix[:, 1], "--", alpha=0.5)
        ax[2].plot(x, attn_mix[:, 0], ".", alpha=0.7, label="words")
        ax[2].plot(x, attn_mix[:, 1], ".", alpha=0.7, label="prosody")
        ax[2].set_ylim([0, 1])
        ax[2].legend()

        if trp_lm is not None:
            ax[-1].plot(trp_lm, "--", label="lm trp")

        for head in range(trp.shape[-1]):
            # ax[-1].bar(x, trp[..., head], label="trp")
            ax[-1].plot(x, trp[..., head], label=f"trp {head}")
        ax[-1].set_ylim([0, 1])
        ax[-1].legend()
        # ax.vlines(sp_idx, ymin=0, ymax=1, color="k", linewidth=2, alpha=0.3)
        ax[-1].set_xticks(torch.arange(len(tokens)))
        ax[-1].set_xticklabels(tokens, rotation=65, fontsize=10)
        ax[-1].set_xlim([0, len(tokens)])

        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def plot_hist(pos, neg, plot=False):
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].hist(
            pos[:, 0].unsqueeze(0), range=(0, 1), bins=100, color="g", label="positive"
        )
        ax[1].hist(
            neg[:, 0].unsqueeze(0), range=(0, 1), bins=100, color="r", label="negative"
        )
        ax[0].legend()
        ax[0].set_ylabel("N")
        ax[1].legend()
        ax[1].set_xlabel("TRP")
        ax[1].set_ylabel("N")
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def plot_gaussian(m, s, m_neg, s_neg, plot=False):
        x = torch.linspace(-0.05, 1.05, 300)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, stats.norm.pdf(x, m, s), "g", label="Positive")
        ax.plot(x, stats.norm.pdf(x, m_neg, s_neg), "r", label="Negative")
        ax.legend()
        if plot:
            plt.pause(0.01)
        plt.tight_layout()
        return fig, ax


class TurnAcoustic(pl.LightningModule):
    def __init__(
        self,
        prosody_encoder=True,
        LM=False,
        prox_input_size=64,
        prox_hidden=64,
        prox_layers=1,
        prox_heads=4,
        prox_horizon=[1],
        prox_horizon_constants=[1],
        prox_resid_pdrop=0.1,
        prox_attn_pdrop=0.1,
        prox_dropout=0.1,
        prox_chunk_size=128,
        prox_loss_constant=1.0,
        prox_model="transformer",
        prosody_frames=100,
        prosody_n_feats=2,
        prosody_hidden=32,
        prosody_layers=3,
        prosody_kernel=5,
        prosody_first_stride=2,
        prosody_n_head=4,
        lm_n_vocab=50259,
        lm_dropout=0.1,
        lm_pretrained="gpt2",
        lm_patience=3,
        modal_n_heads=8,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        learning_rate=1e-4,
        weight_decay=0.01,
        log_rms=True,
        f0_norm=True,
        f0_interpolate=True,
    ):
        super().__init__()
        assert prosody_encoder or LM, "Must use at least one encoder ['prosody', 'LM']"
        self.save_hyperparameters()

        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.log_rms = log_rms
        self.f0_norm = f0_norm
        self.f0_interpolate = f0_interpolate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.example_input_array = [None, None, None]
        self.lm_model = None
        if LM:
            self.lm_model = TurnGPTModel(
                n_vocab=lm_n_vocab,
                dropout=lm_dropout,
                pretrained=lm_pretrained,
            )
            lm_dim = self.lm_model.model.config.n_embd
            if lm_dim != prox_input_size:
                self.lm_postnet = nn.Sequential(
                    nn.Linear(lm_dim, prox_input_size),
                    nn.ReLU(),
                    nn.LayerNorm(prox_input_size),
                )
            else:
                self.lm_postnet = nn.Identity()

            self.lm_has_stopped_training = False
            self.lm_patience = lm_patience
            self.lm_best_loss = torch.tensor(np.Inf)
            self.lm_wait_count = 0
            self.prox_loss_constant = prox_loss_constant
            self.example_input_array[0] = torch.randint(0, 10, (1, 128))
            self.example_input_array[1] = torch.randint(0, 10, (1, 128))

        self.prosody_encoder = None
        if prosody_encoder:
            self.prosody_encoder = ProsodyEncoder(
                frames=prosody_frames,
                n_feats=prosody_n_feats,
                hidden=prosody_hidden,
                layers=prosody_layers,
                kernel=prosody_kernel,
                first_stride=prosody_first_stride,
                output_size=prox_input_size,
                n_head=prosody_n_head,
            )
            self.example_input_array[2] = torch.rand(1, 128, prosody_frames, 2)

        self.modal_mixer = None
        if prosody_encoder and LM:
            self.modal_mixer = Attention1D(
                D=prox_input_size, features=2, features_out=1, num_heads=modal_n_heads
            )

        if prox_model == "rnn":
            self.proximity_model = ProxRNN(
                input_size=prox_input_size,
                hidden_size=prox_hidden,
                n_layer=prox_layers,
                horizon=prox_horizon,
                horizon_constants=prox_horizon_constants,
                dropout=prox_dropout,
            )
        else:
            self.proximity_model = ProxTransformer(
                input_size=prox_input_size,
                hidden_size=prox_hidden,
                n_layer=prox_layers,
                n_head=prox_heads,
                horizon=prox_horizon,
                horizon_constants=prox_horizon_constants,
                resid_pdrop=prox_resid_pdrop,
                attn_pdrop=prox_attn_pdrop,
                dropout=prox_dropout,
                chunk_size=prox_chunk_size,
            )

    def forward(
        self, input_ids=None, speaker_ids=None, prosody=None, output_attentions=False
    ):
        ret = {}
        if self.lm_model is not None and input_ids is not None:
            out = self.lm_model(input_ids, speaker_ids)
            ret["logits"] = out["logits"]
            ret["z_w"] = self.lm_postnet(out["z"])
            z = ret["z_w"]

        if self.prosody_encoder is not None and prosody is not None:
            out = self.prosody_encoder(prosody, output_attentions)
            ret["z_p"] = out["z"]
            z = out["z"]
            if output_attentions:
                ret["attn_pros"] = out["attn_pros"]

        if self.modal_mixer is not None:
            modal_in = torch.stack((ret["z_p"], ret["z_w"]), dim=-2)
            z, attn_mix = self.modal_mixer(modal_in)
            z = z.squeeze(-2)
            ret["z_m"] = z
            if output_attentions:
                ret["attn_mix"] = attn_mix.squeeze(-2).detach().cpu()

        out = self.proximity_model(z, output_attentions)
        ret["prox_logits"] = out["logits"]
        if "attn_prox" in out:
            ret["attn_prox"] = out["attn_prox"]
        return ret

    def _trp_lm(self, logits):
        prob = F.softmax(logits, dim=-1)
        prob = torch.stack((prob[..., self.sp1_idx], prob[..., self.sp2_idx]), dim=-1)
        prob, _ = prob.max(dim=-1)
        return prob

    def _trp(self, logits):
        return torch.sigmoid(logits)  # B, N, 1

    @torch.no_grad()
    def trp(self, input_ids, speaker_ids, prosody=None, output_attentions=False):
        out = self(
            input_ids=input_ids.to(self.device),
            speaker_ids=speaker_ids.to(self.device),
            prosody=prosody.to(self.device),
            output_attentions=output_attentions,
        )
        out["trp"] = self._trp(out["prox_logits"])  # B, N, 1
        if "logits" in out:
            out["trp_lm"] = self._trp_lm(out["logits"])
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )

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
        # shift for labels
        batch["labels"] = batch["input_ids"][:, 1:].contiguous()
        for k, v in batch.items():
            if not k == "labels":
                batch[k] = v[:, :-1].contiguous()

        batch["prosody"] = None
        if "f0" in batch and "rms" in batch:
            batch["prosody"] = torch.stack((batch["f0"], batch["rms"]), dim=-1)

        return batch

    def training_step(self, batch, *args, **kwargs):
        b = self.shared_step(batch)

        out = self(
            input_ids=b["input_ids"], speaker_ids=b["speaker_ids"], prosody=b["prosody"]
        )

        prox_losses = self.proximity_model.loss_function(
            out["prox_logits"],
            b["labels"],
            self.sp1_idx,
            self.sp2_idx,
            self.pad_idx,
        )
        self.log(
            "ploss",
            prox_losses["loss"],
            prog_bar=True,
            logger=True,
        )
        loss = prox_losses["loss"]

        if self.lm_model is not None:
            lm_loss = self.loss_function_lm(out["logits"], b["labels"])
            self.log(
                "lm_loss",
                lm_loss,
                prog_bar=True,
                logger=True,
            )

            # add LM loss and proximity loss
            if self.lm_has_stopped_training:  # only use value of lm_loss
                loss = lm_loss.detach() + self.prox_loss_constant * loss
            else:
                loss = lm_loss + self.prox_loss_constant * loss

            self.log(
                "loss",
                loss,
                logger=True,
            )
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        """for multi-gpu"""
        return {"loss": batch_parts["loss"].mean()}

    def validation_step(self, batch, *args, **kwargs):
        b = self.shared_step(batch)

        out = self(
            input_ids=b["input_ids"], speaker_ids=b["speaker_ids"], prosody=b["prosody"]
        )
        prox_losses = self.proximity_model.loss_function(
            out["prox_logits"],
            b["labels"],
            self.sp1_idx,
            self.sp2_idx,
            self.pad_idx,
        )
        self.log(
            "val_ploss",
            prox_losses["loss"],
            prog_bar=True,
            logger=True,
        )
        loss = prox_losses["loss"]

        ret = {}
        if self.lm_model is not None:
            lm_loss = self.loss_function_lm(out["logits"], b["labels"])
            self.log(
                "val_lm_loss",
                lm_loss,
                prog_bar=True,
                logger=True,
            )
            # add LM loss and proximity loss
            loss = lm_loss + self.prox_loss_constant * loss
            self.log(
                "val_loss",
                loss,
                logger=True,
            )
            ret["lm_loss"] = lm_loss
        ret["val_loss"] = loss
        return ret

    def validation_epoch_end(self, validation_step_outputs):
        """for multi-gpu"""

        if not self.lm_has_stopped_training:
            current = []
            for v in validation_step_outputs:
                current.append(v["lm_loss"])
            current = torch.tensor(current).mean()
            if current < self.lm_best_loss:
                self.lm_best_loss = current
                torch.save(self.lm_model.state_dict(), "/tmp/lm_state_dict.pt")
                self.lm_stop_epoch = self.current_epoch
                self.lm_stop_step = self.global_step
                self.lm_wait_count = 0
            else:
                self.lm_wait_count += 1

            if self.lm_wait_count > self.lm_patience:
                print("Stopped training LM")
                print("lm best loss: ", self.lm_best_loss)
                print("epoch:       ", self.lm_stop_epoch)
                print("global step: ", self.lm_stop_step)
                print("load best lm model")
                self.lm_model.load_state_dict(torch.load("/tmp/lm_state_dict.pt"))
                print("loaded...")
                self.lm_model.freeze()
                print("freeze...")
                self.lm_has_stopped_training = True

    def evaluation(self, dataloader):
        self.eval()  # eval mode

        loss = {}
        pos, neg = [], []
        for batch in tqdm(dataloader, desc="Evaluation"):
            batch = self.shared_step(batch)
            with torch.no_grad():
                out = self.trp(
                    input_ids=batch["input_ids"],
                    speaker_ids=batch["speaker_ids"],
                    prosody=batch["prosody"],
                )
                prox_losses = self.proximity_model.loss_function(
                    out["prox_logits"],
                    batch["labels"].to(self.device),
                    self.sp1_idx,
                    self.sp2_idx,
                    self.pad_idx,
                )
            P, N = get_positive_and_negative_indices(
                batch["input_ids"], self.sp1_idx, self.sp2_idx, self.pad_idx
            )
            pos.append(out["trp"][P].cpu())
            neg.append(out["trp"][N].cpu())
            for k, v in prox_losses.items():
                if "horizon" in k:
                    if k in loss:
                        loss[k].append(v)
                    else:
                        loss[k] = [v]
        for k, v in loss.items():
            loss[k] = torch.tensor(v)
        result = {"pos": torch.cat(pos), "neg": torch.cat(neg)}
        result["hist"] = Plots.plot_hist(result["pos"], result["neg"])
        result["pos_mean"] = result["pos"].mean(dim=0)
        result["pos_std"] = result["pos"].std(dim=0)
        result["neg_mean"] = result["neg"].mean(dim=0)
        result["neg_std"] = result["neg"].std(dim=0)
        return result

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser, LM=False, prosody_encoder=False):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--proximity_model", type=str, default="transformer")
        parser.add_argument(
            "--prosody_encoder", action="store_true", default=prosody_encoder
        )
        parser.add_argument("--LM", action="store_true", default=LM)
        parser.add_argument(
            "--prox_loss_constant",
            type=float,
            default=1.0,
        )

        temp_args, _ = parser.parse_known_args()

        parser = ProxTransformer.add_model_specific_args(parser)
        parser.add_argument("--lm_pretrained", type=str, default="gpt2")
        parser.add_argument("--lm_dropout", type=float, default=0.1)
        parser.add_argument("--lm_patience", type=int, default=3)
        parser = ProsodyEncoder.add_model_specific_args(parser)
        parser.add_argument("--modal_n_heads", type=int, default=2)
        parser.add_argument("--modal_in", type=int, default=64)
        return parser


def main():
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os import environ
    import sys

    from turngpt.acousticDM import AudioDM
    from ttd.utils import get_run_dir

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lm_checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TurnAcoustic.add_model_specific_args(
        parser, prosody_encoder=False, LM=False
    )
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["switchboard"],
        f0=False,
        rms=False,
        waveform=False,
        normalize_f0=False,
        interpolate_f0=False,
    )
    args = parser.parse_args()

    assert (
        args.prosody_encoder or args.LM
    ), "Must use at least one encoder ['--prosody_encoder', '--LM']"

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    args.modal_n_heads = 1
    for prox_input_size in [32, 64, 96]:
        args.prox_input_size = prox_input_size
        for learning_rate in [1e-3, 4e-4, 1e-4]:
            args.learning_rate = learning_rate
            print("prox_input_size: ", args.prox_input_size)
            print("learning_rate: ", args.learning_rate)

            # Initialize Model
            model = get_model(args, lm_n_vocab=len(dm.tokenizer))

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

                name = "TurnAcoustic"
                if args.LM:
                    name += "_LM-" + model.lm_model.__class__.__name__
                if args.prosody_encoder:
                    name += "_P-" + model.prosody_encoder.__class__.__name__
                desc = f"{name} training"
                logger = TensorBoardLogger(
                    args.save_dir,
                    name=name,
                    log_graph=False,
                )
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
            trainer.fit(model, datamodule=dm)


def mess_around():
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lm_checkpoint", type=str, default=None)
    parser = TurnAcoustic.add_model_specific_args(parser)
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
    args.log_rms = True

    # MAKE SURE THAT THE DATA IS CORRECT w.r.t. THE MODEL !!! (log_rms, etc)

    args.batch_size = 8
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()

    args.checkpoint = (
        "checkpoints/TurnAcoustic/f0ni_rmsl/checkpoints/epoch=26-val_loss=0.12482.ckpt"
    )
    args.checkpoint = "checkpoints/TurnAcoustic/version_5/checkpoints/epoch=17-val_loss=0.76582.ckpt"  # log_rms, f0_norm, f0_inter
    args.checkpoint = "TurnGPT/turngpt/runs/TurnAcoustic_LM-TurnGPTModel_P-ProsodyEncoder/version_1/checkpoints/epoch=7-val_loss=3.16107.ckpt"  # log_rms, f0_norm, f0_inter
    # args.checkpoint = "checkpoints/TurnAcoustic/version_3/checkpoints/epoch=26-val_loss=0.12482.ckpt"  # log_rms, f0_norm, f0_inter
    # args.checkpoint = "TurnGPT/turngpt/runs/TurnAcoustic_LM-TurnGPTModel_P-ProsodyEncoder/version_1/checkpoints/epoch=1-val_loss=3.31755.ckpt"
    # args.checkpoint = None
    # args.prosody_encoder = True
    # args.LM = True
    model = get_model(args, lm_n_vocab=len(dm.tokenizer))
    model.eval()

    result = model.evaluation(loader)
    result["hist"][0].savefig("test.png")

    plt.pause(0.01)

    def view_all_sp(out, batch, L=40):
        fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
        sp_b, sp_idx = get_speaker_shift_indices(
            batch["input_ids"], model.sp1_idx, dm.sp2_idx
        )
        tokens = convert_ids_to_tokens(batch["input_ids"], dm.tokenizer)
        for b in range(batch["input_ids"].shape[0]):
            for target in sp_idx[sp_b == b]:
                end = target
                start = end - L
                if start < 0:
                    start = 0
                rel_target = end - start - 1
                if rel_target < 1:
                    continue
                for a in ax:
                    a.cla()
                if "trp_lm" in out:
                    tmp_trp_lm = out["trp_lm"][b, start:end].cpu()
                else:
                    tmp_trp_lm = None
                fig, ax = Plots.plot_turn_acoustic(
                    rel_target,
                    out["trp"][b, start:end].cpu(),
                    out["attn_prox"][b, :, :, start:end, start:end].cpu(),
                    out["attn_pros"][b, start:end].cpu(),
                    out["attn_mix"][b, start:end].cpu(),
                    tokens[b, start:end],
                    trp_lm=tmp_trp_lm,
                    fig=fig,
                    ax=ax,
                )
                input()

    batch = next(iter(loader))
    b = model.shared_step(batch)
    with torch.no_grad():
        out = model.trp(
            b["input_ids"], b["speaker_ids"], b["prosody"], output_attentions=True
        )

    view_all_sp(out, b)

    plt.close("all")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from os.path import split, join

import torch

import pytorch_lightning as pl

from turngpt.models import Attention1D
from turngpt.models.prosody import ProsodyEncoder
from turngpt.models.proximity import ProxTransformer
from turngpt.models.pretrained import TurnGPTModel


def hparams_from_checkpoint(checkpoint):
    return join(split(split(checkpoint)[0])[0], "hparams.yaml")


class TurnAcoustic(pl.LightningModule):
    def __init__(
        self,
        prosody_encoder=True,
        LM=False,
        prox_hidden=128,
        prox_layers=1,
        prox_heads=8,
        prox_horizon=[1],
        prox_horizon_constants=[1],
        prox_resid_pdrop=0.1,
        prox_attn_pdrop=0.1,
        prox_dropout=0.1,
        prox_chunk_size=128,
        prox_loss_constant=1.0,
        prosody_frames=100,
        prosody_n_feats=2,
        prosody_hidden=32,
        prosody_layers=3,
        prosody_kernel=5,
        prosody_first_stride=2,
        prosody_output_size=128,
        prosody_n_head=8,
        lm_n_vocab=50259,
        lm_dropout=0.1,
        lm_pretrained="gpt2",
        modal_n_heads=8,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        learning_rate=1e-4,
        weight_decay=0.01,
    ):
        super().__init__()
        assert prosody_encoder or LM, "Must use at least one encoder ['prosody', 'LM']"
        self.save_hyperparameters()

        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.lm_model = None
        if LM:
            self.lm_model = TurnGPTModel(
                n_vocab=lm_n_vocab,
                dropout=lm_dropout,
                pretrained=lm_pretrained,
            )
            modal_in = self.lm_model.model.config.n_embd
            self.prox_loss_constant = prox_loss_constant

        self.prosody_encoder = None
        if prosody_encoder:
            if not LM:
                modal_in = prosody_output_size

            self.prosody_encoder = ProsodyEncoder(
                frames=prosody_frames,
                n_feats=prosody_n_feats,
                hidden=prosody_hidden,
                layers=prosody_layers,
                kernel=prosody_kernel,
                first_stride=prosody_first_stride,
                output_size=modal_in,
                n_head=prosody_n_head,
            )

        self.modal_mixer = None
        if prosody_encoder and LM:
            self.modal_mixer = Attention1D(
                D=modal_in, features=2, features_out=1, num_heads=modal_n_heads
            )

        self.proximity_model = ProxTransformer(
            input_size=modal_in,
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
            ret["z_w"] = out["z"]
            z = out["z"]

        if self.prosody_encoder is not None and prosody is not None:
            out = self.prosody_encoder(prosody, output_attentions)
            ret["z_p"] = out["z"]
            z = out["z"]
            if "attn_feat" in out:
                ret["attn_feat"] = out["attn_feat"]

        if self.modal_mixer is not None:
            z, attn_mix = self.modal_mixer(
                torch.stack((ret["z_p"], ret["z_w"]), dim=-2)
            )
            z = z.squeeze(-2)
            ret["z_m"] = z
            if output_attentions:
                ret["attn_mix"] = attn_mix.squeeze(-2).detach().cpu()

        out = self.proximity_model(z, output_attentions)
        ret["prox_logits"] = out["logits"]
        if "attn_prox" in out:
            ret["attn_prox"] = out["attn_prox"]
        return ret

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

        return {"val_loss": loss}

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser, LM=False, prosody_encoder=False):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = ProxTransformer.add_model_specific_args(parser)
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
        parser.add_argument("--lm_pretrained", type=str, default="gpt2")
        parser.add_argument("--lm_dropout", type=float, default=0.1)
        parser = ProsodyEncoder.add_model_specific_args(parser)
        parser.add_argument("--modal_n_heads", type=int, default=8)
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

    # Initialize Model
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
            prosody_output_size=args.prosody_output_size,
            prosody_n_head=args.prosody_n_head,
            lm_n_vocab=len(dm.tokenizer),
            lm_dropout=args.lm_dropout,
            lm_pretrained=args.lm_pretrained,
            modal_n_heads=args.modal_n_heads,
            prox_hidden=args.prox_hidden,
            prox_layers=args.prox_layers,
            prox_heads=args.prox_heads,
            prox_horizon=args.prox_horizon,
            prox_horizon_constants=args.prox_horizon_constants,
            prox_resid_pdrop=args.prox_resid_pdrop,
            prox_attn_pdrop=args.prox_attn_pdrop,
            prox_dropout=args.prox_dropout,
            prox_chunk_size=args.prox_chunk_size,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            pad_idx=dm.pad_idx,
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
                print("Assuming load did not work...")
                import sys

                sys.exit(1)
    else:
        # loading entire model from args.checkpoing (and hparams.yaml)
        model = TurnAcoustic.load_from_checkpoint(
            args.checkpoint, hparams=hparams_from_checkpoint(args.checkpoint)
        )

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
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    import sys

    main()

    sys.exit(0)

    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lm_checkpoint", type=str, default=None)
    parser = TurnAcoustic.add_model_specific_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["switchboard"],
        f0=True,
        rms=True,
        waveform=True,
        normalize_f0=True,
        interpolate_f0=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()

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
            prosody_output_size=args.prosody_output_size,
            prosody_n_head=args.prosody_n_head,
            prox_hidden=args.prox_hidden,
            prox_layers=args.prox_layers,
            prox_heads=args.prox_heads,
            prox_horizon=args.prox_horizon,
            prox_horizon_constants=args.prox_horizon_constants,
            prox_resid_pdrop=args.prox_resid_pdrop,
            prox_attn_pdrop=args.prox_attn_pdrop,
            prox_dropout=args.prox_dropout,
            prox_chunk_size=args.prox_chunk_size,
            lm_n_vocab=len(dm.tokenizer),
            lm_dropout=args.lm_dropout,
            lm_pretrained=args.lm_pretrained,
            modal_n_heads=args.modal_n_heads,
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
        model = TurnAcoustic.load_from_checkpoint(
            args.checkpoint, hparams=hparams_from_checkpoint(args.checkpoint)
        )

    batch = next(iter(loader))
    prosody = torch.stack((batch["f0"], batch["rms"]), dim=-1)
    out = model(
        input_ids=batch["input_ids"], speaker_ids=batch["speaker_ids"], prosody=prosody
    )
    print(out.keys())

    for k, v in out.items():
        print(f"{k}: {v.shape}")

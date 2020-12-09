from argparse import ArgumentParser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turngpt.models import Attention1D
from turngpt.models.encoders import SPFConv
from turngpt.transforms import ClassificationLabelTransform


class SelfAttention(nn.Module):
    """
    Source:
        https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(self, hidden, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert hidden % n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(hidden, hidden)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_frames=100,
        hidden=32,
        in_channels=1,
        num_layers=7,
        kernel_size=3,
        first_stride=2,
        activation="ReLU",
    ):
        super().__init__()
        self.input_frames = input_frames
        self.in_channels = in_channels
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.first_stride = first_stride
        self.padding = 0
        self.activation = getattr(nn, activation)

        self.convs = self._build_conv_layers()
        self.output_size = self.calc_conv_out()

    def conv_output_size(self, input_size, kernel_size, padding, stride):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def calc_conv_out(self):
        n = self.conv_output_size(
            self.input_frames, self.kernel_size, self.padding, self.first_stride
        )
        for i in range(self.num_layers - 1):
            n = self.conv_output_size(n, self.kernel_size, self.padding, 1)
        return n

    def _build_conv_layers(self):
        layers = [
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.hidden,
                kernel_size=self.kernel_size,
                stride=self.first_stride,
                padding=self.padding,
            ),
            self.activation(),
        ]
        for i in range(1, self.num_layers):
            layers += [
                nn.Conv1d(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                self.activation(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-2)  # B, T -> B, 1, T
        x = self.convs(x)
        return x


class ProsodyClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=1,
        conv_hidden=64,
        layers=5,
        kernel=3,
        first_stride=2,
        fc_hidden=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50258,
    ):
        super().__init__()
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            num_layers=layers,
            kernel_size=kernel,
            first_stride=first_stride,
            activation=activation,
        )

        fc_in = int(self.conv_encoder.output_size * conv_hidden)

        self.fc = nn.Linear(fc_in, fc_hidden)
        self.head = nn.Linear(fc_hidden, 1)

        self.learning_rate = learning_rate

        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, x):
        x = self.conv_encoder(x)
        x = torch.sigmoid(self.fc(x.flatten(1)))
        x = self.head(x)
        return x

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x = []
        if "f0" in batch:
            x.append(batch["f0"])

        if "rms" in batch:
            x.append(batch["rms"])
        x = torch.stack(x, dim=-1)
        x, y, inp = self.input_to_label(x, batch["input_ids"])
        if x.ndim == 2:
            x = x.squeeze(-2)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)
        return x, y, inp

    def training_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        self.log("val_loss", loss)
        return {"val_loss": loss}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_in_channels", type=int, default=1)
        parser.add_argument("--prosody_conv_hidden", type=int, default=32)
        parser.add_argument("--prosody_conv_layers", type=int, default=5)
        parser.add_argument("--prosody_first_stride", type=int, default=2)
        parser.add_argument("--prosody_kernel", type=int, default=5)
        parser.add_argument("--prosody_fc_hidden", type=int, default=32)
        parser.add_argument("--prosody_activation", type=str, default="ReLU")

        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def logging(args, name="model"):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os import environ
    from os.path import join
    from ttd.utils import get_run_dir

    # Where to save the training
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print("root dir: ", args.save_dir)
    checkpoint_callback = None
    callbacks = None
    logger = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

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
        # tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        # torch.save(dm.tokenizer, tokenizer_path)
        # print("tokenizer saved -> ", tokenizer_path)

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
    return logger, checkpoint_callback, callbacks


def main():
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser = ProsodyClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=False,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
        rms=False,
    )
    args = parser.parse_args()
    args.log_rms = True

    args.early_stopping = True
    args.patience = 5

    for rms in [True, False]:
        args.rms = rms
        dm = AudioDM(args)
        dm.prepare_data()
        dm.setup("fit")

        for chidden in [32, 64, 128]:
            args.prosody_conv_hidden = chidden
            for layers in [3, 5, 7]:
                args.prosody_conv_layers = layers

                name = "ProsConv"
                in_channels = 0
                if args.rms:
                    in_channels += 1
                    name += "_rms"
                if args.f0:
                    in_channels += 1
                    name += "_f0"

                print(name)
                print("conv hidden: ", chidden)
                print("conv layers: ", layers)

                model = ProsodyClassifier(
                    in_frames=args.prosody_frames,
                    in_channels=in_channels,
                    conv_hidden=args.prosody_conv_hidden,
                    layers=args.prosody_conv_layers,
                    kernel=args.prosody_kernel,
                    first_stride=args.prosody_first_stride,
                    fc_hidden=args.prosody_fc_hidden,
                    learning_rate=args.learning_rate,
                    activation=args.prosody_activation,
                    sp1_idx=dm.sp1_idx,
                    sp2_idx=dm.sp2_idx,
                    pad_idx=dm.pad_idx,
                )

                logger, checkpoint_callback, callbacks = logging(args, name=name)

                trainer = pl.Trainer.from_argparse_args(
                    args,
                    logger=logger,
                    checkpoint_callback=checkpoint_callback,
                    callbacks=callbacks,
                )
                trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    # enc = ProsodyClassifier()
    # x = torch.rand((16, 100))
    # y_pred = enc(x)

    main()

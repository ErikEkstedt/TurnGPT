from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from turngpt.models import ConvEncoder
from turngpt.train_utils import logging


class F0Encoder(nn.Module):
    def __init__(
        self,
        in_frames,
        in_channels=2,
        hidden=32,
        kernel_size=[4, 3, 3, 3, 5, 3, 3, 3],
        padding=0,
        strides=[2, 1, 1, 1, 2, 1, 1, 1],
        fc_hidden=256,
        output_size=128,
        activation="ReLU",
    ):
        super().__init__()

        self.convs = ConvEncoder(
            input_frames=in_frames,
            in_channels=in_channels,
            hidden=hidden,
            num_layers=len(kernel_size),
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            activation=activation,
            independent=False,
        )
        self.conv_output_frames = self.convs.output_size
        self.fc = nn.Linear(hidden * self.convs.output_size, fc_hidden)
        self.head = nn.Linear(fc_hidden, output_size)

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.flatten(1))
        x = torch.relu(x)
        x = torch.sigmoid(self.head(x))
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        # acoustic
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--f0_in_channels", type=int, default=2)
        parser.add_argument("--f0_conv_hidden", type=int, default=32)
        parser.add_argument("--f0_fc_hidden", type=int, default=256)
        parser.add_argument("--f0_output_size", type=int, default=64)
        parser.add_argument("--f0_activation", type=str, default="ReLU")
        parser.add_argument(
            "--f0_kernel",
            nargs="*",
            type=int,
            default=[4, 4, 3, 3],
        )
        parser.add_argument(
            "--f0_strides",
            nargs="*",
            type=int,
            default=[2, 2, 1, 1],
        )
        parser.add_argument(
            "--f0_padding",
            nargs="*",
            type=int,
            default=0,
        )
        return parser


class F0Classifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=2,
        hidden=32,
        kernel_size=[4, 2, 3, 3],
        strides=[2, 2, 1, 1],
        padding=0,
        fc_hidden=256,
        encoder_output_size=128,
        activation="ReLU",
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.f0_cnn_encoder = F0Encoder(
            in_frames=in_frames,
            in_channels=in_channels,
            hidden=hidden,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            fc_hidden=fc_hidden,
            output_size=encoder_output_size,
            activation=activation,
        )
        self.head = nn.Linear(encoder_output_size, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.f0_cnn_encoder(x)
        x = self.head(x)
        return x

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred, y.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, *args, **kwargs):
        y_pred = self(batch["x"])
        loss = self.loss_function(y_pred, batch["y"])
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *args, **kwargs):
        y_pred = self(batch["x"])
        loss = self.loss_function(y_pred, batch["y"])
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser = F0Encoder.add_model_specific_args(parser)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def F0ClassifierExperiment(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """

    from audio.dm_prosody import ProsodyClassificationDM

    parser = F0Classifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ProsodyClassificationDM.add_data_specific_args(
        parser,
        # datasets=["switchboard"],
        datasets=["maptask"],
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
    )
    args = parser.parse_args()

    dm = ProsodyClassificationDM(args)
    dm.prepare_data()
    dm.setup("fit")

    base_name = "F0CNN"

    for fc_hidden in [256, 128, 64, 32]:
        args.f0_fc_hidden = fc_hidden
        name = base_name + "_h{args.f0_fc_hidden}"
        print(name)
        print("conv hidden: ", args.f0_conv_hidden)
        print("conv kernel: ", args.f0_kernel)
        print("conv stride: ", args.f0_strides)
        print("fc hidden: ", args.f0_fc_hidden)
        print("encoder out: ", args.f0_output_size)

        model = F0Classifier(
            in_frames=args.prosody_frames,
            in_channels=args.f0_in_channels,
            hidden=args.f0_conv_hidden,
            kernel_size=args.f0_kernel,
            strides=args.f0_strides,
            fc_hidden=args.f0_fc_hidden,
            encoder_output_size=args.f0_output_size,
            activation=args.f0_activation,
            learning_rate=args.learning_rate,
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

    parser = ArgumentParser()
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    F0ClassifierExperiment(parser)

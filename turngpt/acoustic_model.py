from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchaudio.transforms as AT

from turngpt.models.gpt_mini import Block, TransformerConfig
from turngpt.models import Attention1D


def gradient_check_batch(inp, model, n_batch=1):
    opt = torch.optim.SGD(model.parameters(), lr=1)
    opt.zero_grad()
    inp.requires_grad = True
    if inp.grad is not None:
        inp.grad.zero_()

    # forward
    z = model(inp.to(model.device))
    z[n_batch].sum().backward()

    # calc grad
    x_grad = inp.grad.data.abs()
    for i in range(inp.ndim - 1):
        x_grad = x_grad.sum(dim=-1)

    if x_grad.sum() == x_grad[n_batch]:
        print("Gradient Batch Success!")
    else:
        print("Gradient Batch Failure!")


def gradient_check_word_time(inp, model):
    opt = torch.optim.SGD(model.parameters(), lr=1)
    opt.zero_grad()
    inp.requires_grad = True
    if inp.grad is not None:
        inp.grad.zero_()
    N = inp.shape[1]
    target = N // 2
    z = model(inp.to(model.device))
    z[:, target].sum().backward()
    x_grad = inp.grad.data.abs().sum(dim=0)  # sum batches
    x_grad = x_grad.sum(dim=-1)  # sum prosody feats
    if inp.ndim == 4:
        x_grad = x_grad.sum(dim=-1)  # sum T
    if x_grad[target + 1 :].sum() == 0:
        print("Gradient Step Success!")
    else:
        print("Gradient Step Failure!")


class SPFConv(nn.Module):
    def __init__(
        self,
        frames=20,
        n_feats=4,
        hidden=32,
        output_size=128,
        num_layers=3,
        kernel_size=3,
        independent=True,
        activation="ReLU",
    ):
        super().__init__()
        self.n_feats = n_feats
        self.output_size = output_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.independent = independent
        self.out_frames = frames - (2 * num_layers)
        self.activation = getattr(nn, activation)

        if independent:
            self.groups = n_feats
            self.hidden = n_feats * hidden
        else:
            assert hidden is not None, "Must provide `hidden` for joint processing"
            self.groups = 1
            self.hidden = hidden

        self.convs = self._build_layers()
        self.head = nn.Linear(self.out_frames * hidden, output_size)
        self.ln = nn.LayerNorm(output_size)

    def _build_layers(self):
        layers = [
            nn.Conv1d(
                in_channels=self.n_feats,
                out_channels=self.hidden,
                kernel_size=self.kernel_size,
                groups=self.groups,
            ),
            self.activation(),
        ]
        for i in range(1, self.num_layers):
            layers += [
                nn.Conv1d(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    kernel_size=self.kernel_size,
                    groups=self.groups,
                ),
                self.activation(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        B, N, T, n_feats = x.size()

        # M: B*N
        z = x.view(-1, T, n_feats)  # N, B, T, n_feats -> M, T, n_feats
        z = z.permute(0, 2, 1)  # M, T, n_feats -> M, n_feats, T
        z = self.convs(z)  # M, n_feats, T -> M, hidden, T

        # separate channel encodings
        z = torch.stack(
            z.chunk(n_feats, dim=1), dim=1
        )  # M, hidden, T -> M, n_feats, hh, T
        z = z.flatten(-2)  # M, n_feats, hh, T -> M, n_feats, hh*T
        z = self.ln(self.head(z))  # M, n_feats, hh*T -> M, n_feats, hidden
        z = z.view(
            B, N, n_feats, self.output_size
        )  # M, n_feats, hidden -> B, N, n_feats, hidden
        return z  # ready to be attended by transformer


class AcousticConv(pl.LightningModule):
    def __init__(
        self,
        frames=20,
        n_feats=4,
        enc_hidden=32,
        enc_layers=3,
        hidden=768,
        n_head=8,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_head = n_head
        self.feature_encoder = SPFConv(
            frames,
            n_feats=n_feats,
            hidden=enc_hidden,
            output_size=hidden,
            num_layers=enc_layers,
        )
        self.channel_transformer = Attention1D(
            D=hidden, features=n_feats, features_out=1, num_heads=n_head
        )
        self.ln_f = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.feature_encoder(x)
        x, feat_attn = self.channel_transformer(x)
        feat_attn = feat_attn.squeeze(-2)
        x = x.squeeze(-2)
        x = self.ln_f(x)
        return x, feat_attn

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--acoustic_frames", type=int, default=20)
        parser.add_argument("--acoustic_n_feats", type=int, default=4)
        parser.add_argument("--acoustic_enc_hidden", type=int, default=64)
        parser.add_argument("--acoustic_enc_layers", type=int, default=3)
        parser.add_argument("--acoustic_hidden", type=int, default=768)
        return parser


# AcousticConv + Transformer over time
class AcousticTransformer(pl.LightningModule):
    def __init__(
        self,
        frames=20,
        n_feats=4,
        enc_hidden=32,
        enc_layers=3,
        hidden=768,
        n_head=8,
        n_layer=1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        chunk_size=128,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_head = n_head
        self.feature_encoder = SPFConv(
            frames,
            n_feats=n_feats,
            hidden=enc_hidden,
            output_size=hidden,
            num_layers=enc_layers,
        )
        self.channel_transformer = Attention1D(
            D=hidden, features=n_feats, features_out=1, num_heads=n_head
        )

        # Uni-directional Transformer
        self.config = TransformerConfig(
            n_embd=hidden,
            n_head=n_head,
            n_layer=n_layer,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            block_size=chunk_size,
        )
        self.word_transformer = nn.Sequential(
            *[Block(self.config) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.feature_encoder(x)
        x, feat_attn = self.channel_transformer(x)
        feat_attn = feat_attn.squeeze(-2)
        x = x.squeeze(-2)
        x = self.word_transformer(x)
        return x, feat_attn

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--acoustic_frames", type=int, default=20)
        parser.add_argument("--acoustic_n_feats", type=int, default=4)
        parser.add_argument("--acoustic_enc_hidden", type=int, default=32)
        parser.add_argument("--acoustic_enc_layers", type=int, default=3)
        parser.add_argument("--acoustic_hidden", type=int, default=768)
        parser.add_argument("--acoustic_n_head", type=int, default=8)
        parser.add_argument("--acoustic_n_layer", type=int, default=1)
        parser.add_argument("--acoustic_resid_pdrop", type=float, default=0.1)
        parser.add_argument("--acoustic_attn_pdrop", type=float, default=0.1)
        parser.add_argument("--acoustic_chunk_size", type=int, default=128)
        return parser


def dev():
    frames = 20
    n_feats = 4
    x = torch.rand(4, 100, frames, n_feats)

    model = SPFConv(frames=frames, n_feats=n_feats, hidden=32, output_size=256)
    print(model)

    z = model(x)  # 4, 100, 4, 128
    print("x: ", tuple(x.shape))
    print("z: ", tuple(z.shape))
    gradient_check_batch(x, model)
    gradient_check_word_time(x, model)

    model = AcousticTransformer()
    print(model)
    z = model(x)  # 4, 100, 4, 128
    print("x: ", tuple(x.shape))
    print("z: ", tuple(z.shape))
    gradient_check_batch(x, model)
    gradient_check_word_time(x, model)

    model = AcousticConv(enc_hidden=32, hidden=256)
    print(model)
    z, feat_attn = model(x)  # 4, 100, 4, 128
    print("x: ", tuple(x.shape))
    print("z: ", tuple(z.shape))

    gradient_check_batch(x, model)
    gradient_check_word_time(x, model)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from acousticgpt.acousticDM import AcousticGPTDM

    parser = ArgumentParser()
    parser = AcousticGPTDM.add_data_specific_args(parser, datasets=["maptask"])
    parser = AcousticTransformer.add_model_specific_args(parser)
    args = parser.parse_args()
    args.prosody = True

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model = AcousticTransformer(
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

    z = model(xp)
    print("z: ", tuple(z.shape))

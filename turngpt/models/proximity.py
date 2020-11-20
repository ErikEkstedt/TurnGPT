from argparse import ArgumentParser
import torch
import torch.nn as nn


import pytorch_lightning as pl

from turngpt.models.gpt_mini import Block


class ProxRNN(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.1,
        bias=True,
        rnn="LSTM",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        if rnn.lower() == "gru":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )

        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x):
        z, h = self.rnn(x)
        z = self.drop(z)
        return self.head(z)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )

        # Model
        parser.add_argument("--prox_hidden", default=256, type=int)
        parser.add_argument("--prox_layers", default=2, type=int)
        parser.add_argument("--prox_output", default=2, type=int)
        parser.add_argument("--prox_rnn", default="lstm", type=str)
        return parser


class TransformerConfig:
    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class ProxTransformer(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        n_head,
        n_layer,
        resid_pdrop,
        attn_pdrop,
        chunk_size,
    ):
        super().__init__()
        self.input_size = input_size

        if input_size != hidden_size:
            self.pre_layer = nn.Linear(input_size, hidden_size)
        else:
            self.pre_layer = nn.Identity()

        self.config = TransformerConfig(
            n_embd=hidden_size,
            n_head=n_head,
            n_layer=n_layer,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            block_size=chunk_size,
        )

        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])

        # head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )

        # Model
        parser.add_argument("--prox_hidden", default=256, type=int)
        parser.add_argument("--prox_layers", default=2, type=int)
        parser.add_argument("--prox_output", default=2, type=int)
        parser.add_argument("--prox_heads", default=4, type=int)
        parser.add_argument("--prox_resid_pdrop", default=0.1, type=float)
        parser.add_argument("--prox_attn_pdrop", default=0.1, type=float)
        return parser


if __name__ == "__main__":

    from acousticgpt.acoustic_gpt import gradient_check_batch, gradient_check_word_time

    parser = ArgumentParser()
    parser = ProxRNN.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    model = ProxRNN(
        input_size=768,
        hidden_size=args.prox_hidden,
        output_size=args.prox_output,
        num_layers=args.prox_layers,
    )
    print(model)

    inp = torch.rand(4, 128, 768)
    gradient_check_batch(inp, model)
    gradient_check_word_time(inp, model)

    parser = ArgumentParser()
    parser = ProxTransformer.add_model_specific_args(parser)
    args = parser.parse_args()
    args.chunk_size = 128
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model = ProxTransformer(
        input_size=768,
        output_size=args.prox_output,
        hidden_size=args.prox_hidden,
        n_layer=args.prox_layers,
        n_head=args.prox_heads,
        resid_pdrop=args.prox_resid_pdrop,
        attn_pdrop=args.prox_attn_pdrop,
        chunk_size=args.chunk_size,
    )

    print(model)

    inp = torch.rand(4, 128, 768)
    prox_logits = model(inp)

    gradient_check_batch(inp, model)
    gradient_check_word_time(inp, model)

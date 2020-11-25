from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl

from turngpt.models.gpt_mini import Block, TransformerConfig


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


class ProxTransformer(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        horizon=[1, 3],
        n_head=4,
        n_layer=1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        chunk_size=128,
    ):
        super().__init__()
        if isinstance(horizon, int):
            horizon = [horizon]
        self.horizon = horizon
        self.output_size = len(self.horizon)
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
        self.head = nn.Linear(hidden_size, self.output_size, bias=True)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def create_label(self, labels, sp1_idx, sp2_idx):
        proximity_labels = labels == sp1_idx
        if sp1_idx != sp2_idx:
            proximity_labels += labels == sp2_idx

        labs = []
        for tmp_horizon in self.horizon:
            if tmp_horizon > 1:
                tmp_lab = proximity_labels.unfold(
                    dimension=1, step=1, size=tmp_horizon
                ).sum(dim=-1)
                tmp_lab = (tmp_lab > 0).long().detach()
                labs.append(tmp_lab.float())
            elif tmp_horizon == 1:
                labs.append(proximity_labels.float())
        return labs

    def loss_function(self, logits, labels, sp1_idx, sp2_idx=None, pad_idx=None):
        """
        Labels are token indices shifted to the left.
        Create onehot-vector representing turn-shifts in label.
        Unfold the tensor to look ahead N tokens.
        Sum the onehot values in the unfolded window dim and check if any 1s are present.
        A zero value in the labels indicates no proximity of turn-shift.

        :logits:        torch.Tensor (B, N, 2)
        :labels:        torch.Tensor (B, N)
        :N:             int, n_tokens of horizon
        :sp1_idx:       int, speaker 1 token index
        :sp2_idx:       int, speaker 2 token index
        """
        labs = self.create_label(labels, sp1_idx, sp2_idx)
        losses = []
        for head, label in enumerate(labs):
            if pad_idx is not None:
                lab = label[label != pad_idx]
                pred = logits[:, : label.shape[1], head][label != pad_idx]
                losses.append(F.binary_cross_entropy_with_logits(pred, lab))
            else:
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        logits[:, : label.shape[1], head], label.float()
                    )
                )
        return losses

    #         loss_fn = nn.BCEWithLogitsLoss()
    #         labs = self.create_label(labels, sp1_idx, sp2_idx)
    #         losses = []
    #         for head, label in enumerate(labs):
    #             if pad_idx is not None:
    #                 lab = label[label != pad_idx]
    #                 pred = logits[:, : label.shape[1], head][label != pad_idx]
    #                 losses.append(loss_fn(pred, lab))
    #             else:
    #                 losses.append(loss_fn(logits[:, : label.shape[1], head], label.float()))
    #         return losses

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
        parser.add_argument("--prox_chunk_size", default=128, type=int)
        parser.add_argument(
            "--prox_horizon",
            nargs="*",
            type=int,
            default=[1, 3],
        )
        return parser


if __name__ == "__main__":

    # from acousticgpt.acoustic_gpt import gradient_check_batch, gradient_check_word_time

    # parser = ArgumentParser()
    # parser = ProxRNN.add_model_specific_args(parser)
    # args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    # model = ProxRNN(
    #     input_size=768,
    #     hidden_size=args.prox_hidden,
    #     output_size=args.prox_output,
    #     num_layers=args.prox_layers,
    # )
    # print(model)

    # inp = torch.rand(4, 128, 768)
    # gradient_check_batch(inp, model)
    # gradient_check_word_time(inp, model)

    from turngpt.turngpt_dm import TurnGPTDM

    p = ArgumentParser()
    p = TurnGPTDM.add_data_specific_args(p, datasets=["dailydialog"])
    a = p.parse_args()
    a.chunk_size = 128
    for k, v in vars(a).items():
        print(f"{k}: {v}")
    dm = TurnGPTDM(a)
    dm.prepare_data()
    dm.setup("fit")
    sp1_idx = dm.sp1_idx
    sp2_idx = dm.sp2_idx
    pad_idx = dm.pad_idx

    ################################################################################
    # Transformer
    parser = ArgumentParser()
    parser = ProxTransformer.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    model = ProxTransformer(
        input_size=768,
        hidden_size=args.prox_hidden,
        horizon=args.prox_horizon,
        n_layer=args.prox_layers,
        n_head=args.prox_heads,
        resid_pdrop=args.prox_resid_pdrop,
        attn_pdrop=args.prox_attn_pdrop,
        chunk_size=args.prox_chunk_size,
    )
    print(model)
    print(model.horizon)

    batch = next(iter(dm.val_dataloader()))
    # for batch in dm.val_dataloader():
    #     print(batch[0].shape, (batch[0] == pad_idx).sum())

    input_ids = batch[0]
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    prox_logits = model(torch.rand((*input_ids.shape, 768)))
    loss = model.loss_function(prox_logits, labels, sp1_idx, sp2_idx, pad_idx)
    print(loss)

    # model.loss_function(prox_logits, inp[:, :-1], sp1_idx=dm.sp1_idx, sp2_idx=dm.sp2_idx, pad_)
    # gradient_check_batch(inp, model)
    # gradient_check_word_time(inp, model)

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


def dev_rnn():
    from turngpt.models import gradient_check_batch, gradient_check_word_time

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
        dropout=0.1,
        chunk_size=128,
        horizon_constants=[1, 1],
    ):
        super().__init__()
        if isinstance(horizon, int):
            horizon = [horizon]
        self.horizon = horizon
        self.output_size = len(self.horizon)
        self.input_size = input_size
        self.horizon_constants = horizon_constants
        self.chunk_size = chunk_size
        assert len(horizon) == len(
            horizon_constants
        ), "horizon and horizon constant must be of same length!"

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
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.chunk_size, hidden_size))
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])

        # head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.output_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def embedding(self, x):
        B, N, D = x.size()
        assert N <= self.chunk_size, "Cannot forward, model chunk size is exhausted."

        # forward the GPT model
        # each position maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :N]
        x = x + position_embeddings
        return self.dropout(x)

    def forward(self, x, output_attention=False):
        x = self.pre_layer(x)
        x = self.embedding(x)
        if output_attention:
            self._set_attention()
        x = self.blocks(x)
        x = self.head(self.ln_f(x))
        out = {"logits": x}
        if output_attention:
            out["attn_prox"] = self._get_attention()
        return out

    def _set_attention(self):
        for block in self.blocks:
            block.output_attention = True

    def _unset_attention(self):
        for block in self.blocks:
            block.output_attention = False

    def _get_attention(self):
        attn = []
        for block in self.blocks:
            _attn = block.attn.attention
            if _attn is not None:
                attn.append(_attn)
        return torch.stack(attn, dim=1)

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
                tmp_lab = (tmp_lab > 0).long()
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
        losses = {}
        loss = 0
        for head, (horizon, label) in enumerate(zip(self.horizon, labs)):
            if pad_idx is not None:
                lab = label[label != pad_idx]
                pred = logits[:, : label.shape[1], head][label != pad_idx]
                tmp_loss = F.binary_cross_entropy_with_logits(pred, lab.float())
            else:
                tmp_loss = F.binary_cross_entropy_with_logits(
                    logits[:, : label.shape[1], head], label
                )
            loss += self.horizon_constants[head] * tmp_loss
            losses[f"horizon_{horizon}"] = tmp_loss.detach().item()
        losses["loss"] = loss
        return losses

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )

        # Model
        parser.add_argument("--prox_input_size", default=64, type=int)
        parser.add_argument("--prox_hidden", default=64, type=int)
        parser.add_argument("--prox_layers", default=1, type=int)
        parser.add_argument("--prox_heads", default=4, type=int)
        parser.add_argument("--prox_resid_pdrop", default=0.1, type=float)
        parser.add_argument("--prox_attn_pdrop", default=0.1, type=float)
        parser.add_argument("--prox_dropout", default=0.1, type=float)
        parser.add_argument("--prox_chunk_size", default=128, type=int)
        parser.add_argument(
            "--prox_horizon",
            nargs="*",
            type=int,
            default=[1],
        )
        parser.add_argument(
            "--prox_horizon_constants",
            nargs="*",
            type=int,
            default=[1],
        )
        return parser


class ProxRNN(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        horizon=[1, 3],
        n_layer=1,
        dropout=0.1,
        horizon_constants=[1, 1],
    ):
        super().__init__()
        if isinstance(horizon, int):
            horizon = [horizon]
        self.horizon = horizon
        self.output_size = len(self.horizon)
        self.input_size = input_size
        self.horizon_constants = horizon_constants
        assert len(horizon) == len(
            horizon_constants
        ), "horizon and horizon constant must be of same length!"

        # transformer
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=n_layer, dropout=dropout)

        # head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, self.output_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, output_attention=False):
        x, h = self.rnn(x)
        x = self.head(self.ln_f(x))
        out = {"logits": x}
        return out

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
                tmp_lab = (tmp_lab > 0).long()
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
        losses = {}
        loss = 0
        for head, (horizon, label) in enumerate(zip(self.horizon, labs)):
            if pad_idx is not None:
                lab = label[label != pad_idx]
                pred = logits[:, : label.shape[1], head][label != pad_idx]
                tmp_loss = F.binary_cross_entropy_with_logits(pred, lab.float())
            else:
                tmp_loss = F.binary_cross_entropy_with_logits(
                    logits[:, : label.shape[1], head], label
                )
            loss += self.horizon_constants[head] * tmp_loss
            losses[f"horizon_{horizon}"] = tmp_loss.detach().item()
        losses["loss"] = loss
        return losses

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )

        # Model
        parser.add_argument("--prox_input_size", default=64, type=int)
        parser.add_argument("--prox_hidden", default=64, type=int)
        parser.add_argument("--prox_layers", default=1, type=int)
        parser.add_argument("--prox_dropout", default=0.1, type=float)
        parser.add_argument(
            "--prox_horizon",
            nargs="*",
            type=int,
            default=[1],
        )
        parser.add_argument(
            "--prox_horizon_constants",
            nargs="*",
            type=int,
            default=[1],
        )
        return parser


if __name__ == "__main__":

    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser = ProxTransformer.add_model_specific_args(parser)
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
    batch = next(iter(loader))

    args.prox_layers = 4
    model = ProxTransformer(
        input_size=768,
        hidden_size=args.prox_hidden,
        n_layer=args.prox_layers,
        n_head=args.prox_heads,
        horizon=args.prox_horizon,
        horizon_constants=args.prox_horizon_constants,
        resid_pdrop=args.prox_resid_pdrop,
        attn_pdrop=args.prox_attn_pdrop,
        chunk_size=args.prox_chunk_size,
    )
    print(model)

    batch = next(iter(dm.val_dataloader()))

    x = torch.randn((4, 128, 768))
    o = model(x, True)
    print(o.keys())
    print(o["logits"].shape)
    if "attn_prox" in o:
        print(o["attn_prox"].shape)

    input_ids = batch[0]
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    prox_logits = model(torch.rand((*input_ids.shape, 768)))
    loss = model.loss_function(prox_logits, labels, sp1_idx, sp2_idx, pad_idx)
    print(loss)

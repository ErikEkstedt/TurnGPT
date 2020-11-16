from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F

from turngpt.pl_modules import TurnGPT


class RNN(nn.Module):
    def __init__(self, n_vocab, n_embd, n_layer, dropout=0.1, rnn="lstm"):
        super().__init__()

        if rnn.lower() == "lstm":
            rnn = nn.LSTM
        elif rnn.lower() == "gru":
            rnn = nn.GRU
        else:
            raise NotImplementedError(f"Not a valid network type: {rnn}")

        self.embedding = nn.Embedding(n_vocab, n_embd)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = rnn(n_embd, n_embd, num_layers=n_layer, batch_first=True)
        self.head = nn.Linear(n_embd, n_vocab)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, speaker_ids=None):

        emb = self.embedding(idx)
        if speaker_ids is not None:
            emb += self.embedding(speaker_ids)
        emb = self.dropout(emb)
        z, h = self.rnn(emb)
        output = {"z": z}
        output["logits"] = self.head(z)
        return output


class TurnGPTRnn(TurnGPT):
    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        chunk_size: int = 512,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        dropout: float = 0.1,
        rnn: str = "lstm",
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.pad_idx = pad_idx
        self.lr = lr

        self.model = RNN(n_vocab, n_embd, n_layer, dropout, rnn)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--n_embd", default=768, type=int)
        parser.add_argument("--n_layer", default=3, type=int)
        parser.add_argument("--rnn", default="LSTM", type=str)
        parser.add_argument("--dropout", default=0.3, type=float)
        parser.add_argument("--chunk_size", default=128, type=int)

        # Training
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser

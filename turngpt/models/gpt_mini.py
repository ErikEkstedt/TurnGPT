"""
SOURCE:

https://github.com/karpathy/minGPT
"""

import math
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn import functional as F

from turngpt.pl_modules import TurnGPT

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    use_speaker_emb = False
    predict_speaker_emb = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
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
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.use_speaker_emb = config.use_speaker_emb
        self.predict_speaker_emb = config.use_speaker_emb

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if self.predict_speaker_emb:
            self.speaker_head = nn.Linear(config.n_embd, 2, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_size(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def embedding(self, idx, speaker_ids=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        total_emb = token_embeddings + position_embeddings

        if self.use_speaker_emb and speaker_ids is not None:
            speaker_embeddings = self.tok_emb(speaker_ids)
            total_emb += speaker_embeddings

        x = self.drop(total_emb)
        return x

    def transformer(self, idx, speaker_ids=None):
        x = self.embedding(idx, speaker_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

    def forward(self, idx, speaker_ids=None):
        z = self.transformer(idx, speaker_ids)
        output = {"z": z}
        output["logits"] = self.head(z)
        return output


class TurnGPTMini(TurnGPT):
    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        chunk_size: int = 512,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
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

        self.model_config = GPTConfig(
            vocab_size=n_vocab,
            block_size=chunk_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        self.model = GPT(self.model_config)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--n_embd", default=256, type=int)
        parser.add_argument("--n_head", default=8, type=int)
        parser.add_argument("--n_layer", default=8, type=int)
        parser.add_argument("--chunk_size", default=512, type=int)

        # Training
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser


if __name__ == "__main__":

    from ttd.tokenizer import load_turngpt_tokenizer

    tokenizer = load_turngpt_tokenizer()

    config = GPT1Config(
        vocab_size=len(tokenizer), block_size=256, n_embd=256, n_head=8, n_layer=8
    )
    model = GPT(config)

    print("model: ", model.get_size())

    input_ids = torch.tensor(
        tokenizer.encode(
            " hello there everybody my name is erik and i program for a living"
        )
    ).unsqueeze(0)
    speaker_ids = torch.tensor(
        tokenizer.encode(["<speaker1>"] * x.shape[-1])
    ).unsqueeze(0)

    y = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    speaker_ids = speaker_ids[:, :-1]
    out = model(input_ids, speaker_ids, targets=y)

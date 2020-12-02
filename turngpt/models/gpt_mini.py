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

logger = logging.getLogger(__name__)


class TransformerConfig:
    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


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
    """ https://github.com/karpathy/minGPT """

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

        self.attention = None  # dummy for storing att

    def forward(self, x, output_attention=False):
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

        if output_attention:
            self.attention = att.detach().cpu()

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

        self.output_attention = False

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.output_attention)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(
        self,
        n_vocab,
        n_embd,
        n_head,
        n_layer,
        embd_pdrop,
        resid_pdrop,
        attn_pdrop,
        use_speaker_emb,
        chunk_size,
    ):
        super().__init__()

        self.config = GPTConfig(
            vocab_size=n_vocab,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            use_speaker_emb=use_speaker_emb,
            block_size=chunk_size,
        )
        self.block_size = self.config.block_size
        self.use_speaker_emb = use_speaker_emb
        self.n_embd = self.config.n_embd
        self.n_layer = self.config.n_layer
        self.n_head = self.config.n_head

        # input embedding stem
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_size(self):
        return sum(p.numel() for p in self.parameters())

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
        return torch.stack(attn)

    def body(self, input_ids, speaker_ids, **kwargs):
        x = self.embedding(input_ids, speaker_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

    def forward(self, idx, speaker_ids=None, output_attention=False):
        if output_attention:
            self._set_attention()
        z = self.body(idx, speaker_ids)
        output = {"z": z}
        output["logits"] = self.head(z)
        if output_attention:
            output["attn"] = self._get_attention()
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(
            parents=[parent_parser], conflict_handler="resolve", add_help=False
        )

        # Model
        parser.add_argument("--n_embd", default=256, type=int)
        parser.add_argument("--n_head", default=8, type=int)
        parser.add_argument("--n_layer", default=8, type=int)
        parser.add_argument("--embd_pdrop", default=0.1, type=float)
        parser.add_argument("--resid_pdrop", default=0.1, type=float)
        parser.add_argument("--attn_pdrop", default=0.1, type=float)
        parser.add_argument("--use_speaker_emb", default=True, type=bool)
        parser.add_argument("--chunk_size", default=512, type=int)
        return parser


if __name__ == "__main__":

    from ttd.tokenizer import load_turngpt_tokenizer

    tokenizer = load_turngpt_tokenizer()

    model = GPT(
        n_vocab=len(tokenizer),
        n_embd=256,
        n_head=8,
        n_layer=4,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        use_speaker_emb=True,
        chunk_size=128,
    )

    print("model: ", model.get_size())

    input_ids = torch.tensor(
        tokenizer.encode(
            " hello there everybody my name is erik and i program for a living"
        )
    ).unsqueeze(0)
    speaker_ids = torch.tensor(
        tokenizer.encode(["<speaker1>"] * input_ids.shape[-1])
    ).unsqueeze(0)

    y = input_ids[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    speaker_ids = speaker_ids[:, :-1]

    input_ids = torch.cat([input_ids] * 4)
    speaker_ids = torch.cat([speaker_ids] * 4)

    out = model(input_ids, speaker_ids)
    print(out.keys())

    out = model(input_ids, speaker_ids, output_attention=True)
    print(out.keys())
    print(out["attn"].shape)

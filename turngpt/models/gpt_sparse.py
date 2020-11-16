import math
import logging

from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F

from nupic.torch.modules import KWinners, SparseWeights

from pl_modules import TurnGPT

logger = logging.getLogger(__name__)

"""
def kwinners(x, duty_cycles, k, boost_strength, break_ties=False, relu=False, inplace=False)
relu ?  in function... role?
"""


class SparseCausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(
        self,
        n_head=4,
        hidden=32,
        weight_sparsity=0.9,
        attn_pdrop=0,
        resid_pdrop=0,
        block_size=512,
    ):
        super().__init__()
        assert hidden % n_head == 0
        # key, query, value projections for all heads
        self.key = SparseWeights(nn.Linear(hidden, hidden), sparsity=weight_sparsity)
        self.query = SparseWeights(nn.Linear(hidden, hidden), sparsity=weight_sparsity)
        self.value = SparseWeights(nn.Linear(hidden, hidden), sparsity=weight_sparsity)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(hidden, hidden)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
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


class SparseBlock(nn.Module):
    """ an unassuming (HTM - numenta - Sparse) Transformer block """

    def __init__(
        self,
        hidden=32,
        n_head=4,
        attn_pdrop=0,
        resid_pdrop=0,
        block_size=512,
        weight_sparsity=0.9,
        percent_on=0.1,
        boost_strength=0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.attn = SparseCausalSelfAttention(
            n_head=n_head,
            hidden=hidden,
            weight_sparsity=weight_sparsity,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            block_size=block_size,
        )
        self.mlp = nn.Sequential(
            SparseKWLinear(
                hidden,
                4 * hidden,
                sparsity=weight_sparsity,
                percent_on=percent_on,
                boost_strength=boost_strength,
            ),
            nn.GELU(),
            SparseKWLinear(
                4 * hidden,
                hidden,
                sparsity=weight_sparsity,
                percent_on=percent_on,
                boost_strength=boost_strength,
            ),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SparseKWLinear(nn.Module):
    def __init__(self, input_size, hidden, sparsity, percent_on, boost_strength):
        super().__init__()
        self.sparse_linear = SparseWeights(
            nn.Linear(input_size, hidden), sparsity=sparsity
        )
        self.kw = KWinners(
            n=hidden, percent_on=percent_on, boost_strength=boost_strength
        )

    def forward(self, x):
        z = self.sparse_linear(x)
        z = self.kw(z.view(-1, z.shape[-1])).view(z.size())
        return z


class SparseGPT(nn.Module):
    def __init__(
        self,
        n_vocab,
        n_embd=768,
        n_head=8,
        n_layer=4,
        chunk_size=512,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        weight_sparsity=0.2,
        boost_strength=1.0,
        percent_on=0.2,
    ):
        super().__init__()
        self.block_size = chunk_size

        # input embedding stem
        self.tok_emb = nn.Embedding(n_vocab, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                SparseBlock(
                    hidden=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    block_size=chunk_size,
                    weight_sparsity=weight_sparsity,
                    percent_on=percent_on,
                    boost_strength=boost_strength,
                )
                for _ in range(n_layer)
            ]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_vocab, bias=False)

        # self.apply(self._init_weights)

        logger.info("number of parameters: %e", self.get_size())

    def _init_weights(self, module):
        """
        Are the sparse layer initialized? If so how or If not how should they?
        """
        raise NotImplementedError(
            "How should SparseWeights, KWinners and Linear be initialize?"
        )

    def get_size(self):
        return sum(p.numel() for p in self.parameters())

    def embedding(self, idx, speaker_ids=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        total_emb = token_embeddings + position_embeddings

        if speaker_ids is not None:
            speaker_embeddings = self.tok_emb(speaker_ids)
            total_emb += speaker_embeddings

        x = self.drop(total_emb)
        return x

    def transformer(self, idx, speaker_ids=None):
        x = self.embedding(idx, speaker_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

    def forward(
        self, idx, speaker_ids=None, labels=None, speaker_labels=None, pad_idx=None
    ):
        z = self.transformer(idx, speaker_ids)
        # Language Modelling
        logits = self.head(z)
        output = {"logits": logits, "z": z}
        return output


def plot_sparsity(sparse_layer, figsize=(9, 9), plot=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(sparse_layer.weight.data, aspect="auto", origin="lower")
    ax.set_title("Weight Sparsity Linear")
    ax.set_ylabel("hidden")
    ax.set_xlabel("input")
    if plot:
        plt.pause(0.00001)
    return fig, ax


class TurnGPTSparse(TurnGPT):
    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        chunk_size: int = 512,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 4,
        weight_sparsity: float = 0.2,
        percent_on: float = 0.1,
        boost_strength: float = 1.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.model = SparseGPT(
            n_vocab,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            chunk_size=chunk_size,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            weight_sparsity=weight_sparsity,
            boost_strength=boost_strength,
            percent_on=percent_on,
        )
        self.pad_idx = pad_idx
        self.lr = lr

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--chunk_size", default=512, type=int)
        parser.add_argument("--n_embd", default=768, type=int)
        parser.add_argument("--n_head", default=8, type=int)
        parser.add_argument("--n_layer", default=12, type=int)
        parser.add_argument(
            "--attn_pdrop", default=0
        )  # no dropout with kwinners + sparsity
        parser.add_argument(
            "--resid_pdrop", default=0
        )  # no dropout with kwinners + sparsity
        parser.add_argument(
            "--embd_pdrop", default=0
        )  # no dropout with kwinners + sparsity
        parser.add_argument(
            "--weight_sparsity", default=0.2, type=float
        )  #  % of weights that are zero in the layer.)
        parser.add_argument("--percent_on", type=float, default=0.1)
        parser.add_argument("--boost_strength", type=float, default=1.0)
        return parser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = SparseGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    args.vocab_size = 52057
    model = SparseGPT(args)

    device = "cuda"
    model = model.to(device)

    N = 4
    T = 20
    input_ids = torch.randint(0, args.vocab_size, (4, T)).to(device)
    print("input_ids: ", tuple(input_ids.shape))
    out = model(input_ids)
    print("logits: ", out["logits"].shape, out["logits"].sum())

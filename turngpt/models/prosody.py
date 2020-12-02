from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchaudio.transforms as AT

from turngpt.models.encoders import SPFConv
from turngpt.models import Attention1D


def plot_prox(
    prox_logits, input_ids, tokenizer, sp1_idx, sp2_idx, attn_feat=None, plot=False
):
    inp_tokens = convert_ids_to_tokens(input_ids, tokenizer)[0]
    _, sp_idx = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)

    probs = []
    for head in range(prox_logits.shape[-1]):
        probs.append(torch.sigmoid(prox_logits[..., head]).cpu())

    if attn_feat is not None:
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        b = ax[0]
        a = ax[-1]
        b.plot(attn_feat[:, 0], label="f0")
        b.vlines(sp_idx, ymin=0, ymax=1, color="k", linewidth=2, alpha=0.3)
        b.legend()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), sharex=True)
        a = ax
    # prox
    for i, p in enumerate(probs):
        a.plot(p, label=f"prox {i}", alpha=0.6)
    a.vlines(sp_idx, ymin=0, ymax=1, color="k", linewidth=2, alpha=0.3)
    a.set_xticks(torch.arange(len(inp_tokens)))
    a.set_xticklabels(inp_tokens, rotation=65, fontsize=10)
    a.set_xlim([0, len(inp_tokens)])

    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


class ProsodyEncoder(nn.Module):
    def __init__(
        self,
        frames=100,
        n_feats=2,
        hidden=32,
        layers=3,
        kernel=3,
        first_stride=2,
        output_size=128,
        n_head=8,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_head = n_head
        self.n_feats = n_feats
        self.feature_encoder = SPFConv(
            frames,
            n_feats=n_feats,
            hidden=hidden,
            kernel_size=kernel,
            output_size=output_size,
            first_stride=first_stride,
            num_layers=layers,
        )

        if n_feats > 1:
            self.channel_transformer = Attention1D(
                D=output_size, features=n_feats, features_out=1, num_heads=n_head
            )

    @property
    def device(self):
        return self.feature_encoder.device

    def plot_attention(self, attn, feats=["f0", "rms"], ax=None):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for i, feat in enumerate(feats):
            ax.plot(attn[..., i], label=feat)
        ax.set_ylim([0, 1])
        return fig, ax

    def forward(self, x, output_attention=False):
        ret = {}
        x = self.feature_encoder(x)
        if self.n_feats > 1:
            x, feat_attn = self.channel_transformer(x)
            x = x.squeeze(-2)
            if output_attention:
                ret["attn_pros"] = feat_attn.squeeze(-2).detach().cpu()
        else:
            x = x.squeeze(-2)
        ret["z"] = x
        return ret

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_n_feats", type=int, default=2)
        parser.add_argument("--prosody_hidden", type=int, default=32)
        parser.add_argument("--prosody_output_size", type=int, default=128)
        parser.add_argument("--prosody_layers", type=int, default=3)
        parser.add_argument("--prosody_kernel", type=int, default=5)
        parser.add_argument("--prosody_n_head", type=int, default=8)
        parser.add_argument("--prosody_first_stride", type=int, default=2)
        return parser


if __name__ == "__main__":

    # debug
    import matplotlib.pyplot as plt
    from turngpt.acousticDM import AudioDM
    from ttd.tokenizer_helpers import convert_ids_to_tokens
    from turngpt.turngpt_utils import get_speaker_shift_indices
    from turngpt.models import gradient_check_batch, gradient_check_word_time

    parser = ArgumentParser()
    parser = ProsodyEncoder.add_model_specific_args(parser)
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

    enc = ProsodyEncoder(
        frames=args.prosody_frames,
        n_feats=args.prosody_n_feats,
        hidden=args.prosody_hidden,
        layers=args.prosody_layers,
        kernel=args.prosody_kernel,
        first_stride=args.prosody_first_stride,
        output_size=args.prosody_output_size,
        n_head=args.prosody_n_head,
    )

    batch = next(iter(dm.val_dataloader()))
    x = torch.stack((batch["f0"], batch["rms"]), dim=-1)
    o = enc(x, output_attention=True)
    for k, v in o.items():
        print(f"{k}: {tuple(v.shape)}")

    gradient_check_batch(x, enc)
    gradient_check_word_time(x, enc)

    fig, ax = enc.plot_attention(o["attn_pros"][0])
    plt.show()

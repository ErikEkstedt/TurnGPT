from argparse import ArgumentParser
from os.path import join, basename
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from librosa import time_to_samples
from librosa.feature import rms

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT


from turngpt.models.pretrained import TurnGPTModel
from turngpt.F0 import F0, f0_z_normalize, interpolate_forward, F0_swipe

from ttd.utils import read_json
from ttd.tokenizer import (
    load_turngpt_tokenizer,
    get_special_tokens_dict,
)
from ttd.tokenizer_helpers import (
    tokenize_string,
    convert_ids_to_tokens,
    remove_punctuation_capitalization,
)
from ttd.vad_helpers import vad_from_word_level, percent_to_onehot


DATA = "data/sample_phrases"


def modifiy_state_dict_to_turngptmodel(checkpoint_path):
    from copy import deepcopy
    from collections import OrderedDict
    import re

    check = torch.load(checkpoint_path)
    new_check = deepcopy(check)
    new_state_dict = OrderedDict()
    for k, v in check["state_dict"].items():
        new_k = re.sub("^lm_model\.", "", k)
        new_state_dict[new_k] = v
    new_check["state_dict"] = new_state_dict
    new_path = checkpoint_path.replace(".ckpt", "_V2.ckpt")
    torch.save(new_check, new_path)


@torch.no_grad()
def TRP(logits, sp1_idx, sp2_idx):
    probs = F.softmax(logits, dim=-1)
    trp, _ = probs[:, :, (sp1_idx, sp2_idx)].max(dim=-1)
    return trp


def plot_trp(trp, tokens, ax=None, figsize=(9, 6), plot=True):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    # ax.plot(trp, label='lm trp')
    ax.bar(torch.arange(len(trp)), trp, label="lm trp")
    ax.set_xticks(torch.arange(len(trp)))
    ax.set_xticklabels(tokens, rotation=65, fontsize=10)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left")
    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


def extract_phrase_data(dset, f0_normalize, f0_interpolate, pad_idx, f0_smooth=False):
    data = []
    f0 = []
    voiced = []
    for d in dset:
        valid = d["input_ids"] != pad_idx
        pitch = d["f0"][valid]
        v = pitch != 0
        f0.append(pitch)
        voiced.append(v)

        new_d = {
            "input_ids": d["input_ids"][valid],
            "speaker_ids": d["input_ids"][valid],
            "f0": pitch,
            "waveform": d["waveform"][valid],
            "voiced": v,
        }
        if "rms" in d:
            new_d["rms"] = d["rms"][valid]
        data.append(new_d)

    if f0_normalize:

        f = torch.cat(f0)
        voiced = f != 0
        m = f[voiced].mean()
        s = f[voiced].std()
        for d in data:
            for i, segment in enumerate(d["f0"]):
                d["f0"][i] = f0_z_normalize(d["f0"][i], mean=m, std=s)

    if f0_interpolate:
        for d in data:
            for i, segment in enumerate(d["f0"]):
                d["f0"][i] = interpolate_forward(d["f0"][i], d["voiced"][i])

    if f0_smooth:
        pass

    return data


def prepare():
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        # datasets=["maptask"],
        datasets=["samplephrases"],
        f0=True,
        waveform=True,
        f0_normalize=False,
        f0_interpolate=False,
        f0_smooth=False,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    # Hacky data
    data = extract_phrase_data(
        dm.val_dset, f0_normalize=True, f0_interpolate=True, pad_idx=dm.pad_idx
    )

    # Load Model And extract trp text
    print("load model")
    args.checkpoint = (
        "checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981_V2.ckpt"
    )
    model = TurnGPTModel.load_from_checkpoint(checkpoint_path=args.checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    lm_trps = []
    print("get trp")
    with torch.no_grad():
        for d in data:
            inp = d["input_ids"].to(model.device).unsqueeze(0)
            sp = d["speaker_ids"].to(model.device).unsqueeze(0)
            out = model(inp, sp)
            t = TRP(out["logits"], dm.sp1_idx, dm.sp2_idx).cpu()
            d["lm_trp"] = t.view(-1, 1)

    torch.save(
        {"data": data, "tokenizer": dm.tokenizer}, "checkpoints/phrase_data_f0n_f0i.pt"
    )
    print("saved data")


def append_turn_acoustic(data):
    from turngpt.TurnAcoustic import TurnAcoustic, get_model

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lm_checkpoint", type=str, default=None)
    parser = TurnAcoustic.add_model_specific_args(parser)
    args = parser.parse_args()
    args.log_rms = True
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")

    # MAKE SURE THAT THE DATA IS CORRECT w.r.t. THE MODEL !!! (log_rms, etc)
    args.checkpoint = (
        "checkpoints/TurnAcoustic/f0ni_rmsl/checkpoints/epoch=26-val_loss=0.12482.ckpt"
    )
    args.checkpoint = "checkpoints/TurnAcoustic/version_5/checkpoints/epoch=17-val_loss=0.76582.ckpt"  # log_rms, f0_norm, f0_inter
    args.checkpoint = "TurnGPT/turngpt/runs/TurnAcoustic_LM-TurnGPTModel_P-ProsodyEncoder/version_1/checkpoints/epoch=7-val_loss=3.16107.ckpt"  # log_rms, f0_norm, f0_inter
    # args.checkpoint = "checkpoints/TurnAcoustic/version_3/checkpoints/epoch=26-val_loss=0.12482.ckpt"  # log_rms, f0_norm, f0_inter
    # args.checkpoint = "TurnGPT/turngpt/runs/TurnAcoustic_LM-TurnGPTModel_P-ProsodyEncoder/version_1/checkpoints/epoch=1-val_loss=3.31755.ckpt"
    # args.checkpoint = None
    # args.prosody_encoder = True
    # args.LM = True
    model = get_model(args, lm_n_vocab=50259)
    model.eval()

    def shared_step(batch):
        # shift for labels
        batch["labels"] = batch["input_ids"][:, 1:].contiguous()
        for k, v in batch.items():
            if not k == "labels":
                batch[k] = v[:, :-1].contiguous()

        batch["prosody"] = None
        if "f0" in batch and "rms" in batch:
            batch["prosody"] = torch.stack((batch["f0"], batch["rms"]), dim=-1)

    ta_results = []
    with torch.no_grad():
        for d in data:
            inp = d["input_ids"].to(model.device).unsqueeze(0)
            sp = d["speaker_ids"].to(model.device).unsqueeze(0)
            prosody = torch.stack((d["f0"], d["rms"]), dim=-1).unsqueeze(0)
            out = model.trp(inp, sp, prosody)
            d["trp_ta"] = out["trp"].cpu().squeeze(0)
            d["trp_ta_lm"] = out["trp_lm"].cpu().squeeze(0)
    return data


def append_prosody_gpt(data):
    from turngpt.prosody_gpt import ProsodyGPT

    parser = ArgumentParser()
    parser = ProsodyGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")

    checkpoint = (
        "TurnGPT/turngpt/runs/PGPT/version_2/checkpoints/epoch=13-val_loss=0.21682.ckpt"
    )
    # checkpoint = "checkpoints/PGPT/version_0/checkpoints/epoch=10-val_loss=0.13770.ckpt"
    hparams = "TurnGPT/turngpt/runs/PGPT/version_2/hparams.yaml"
    model = ProsodyGPT.load_from_checkpoint(checkpoint, hparams_file=hparams)
    # model = ProsodyGPT.load_from_checkpoint(checkpoint)
    model.eval()

    d = data[0]
    for d in data:
        with torch.no_grad():
            # shared_step
            x = []
            if "f0" in d:
                x.append(d["f0"].unsqueeze(0))
            if "rms" in d:
                x.append(d["rms"].unsqueeze(0))
            x = torch.stack(x, dim=-2)
            x, y, inp = model.input_to_label(x, d["input_ids"].unsqueeze(0))
            y_pred, vq_loss = model(inp.to(model.device), x.to(model.device))
            emb, enc, _ = model.encode(inp.to(model.device), x.to(model.device))
            loss = model.loss_function(y_pred, y.to(model.device))
            loss += vq_loss
            probs = torch.sigmoid(y_pred).cpu()
            d["trp_pros_gpt"] = probs.view(-1, 1)
    return data


def append_pros_conv(data):
    from turngpt.prosody_classifier import ProsodyClassifier

    checkpoint = (
        "checkpoints/ProsClassifier/version_4/checkpoints/epoch=2-val_loss=0.47783.ckpt"
    )
    model = ProsodyClassifier.load_from_checkpoint(
        checkpoint,
        in_channels=2,
        layers=7,
        first_stride=[2, 1, 1, 1, 1, 1, 1],
        kernel=[5, 5, 5, 5, 5, 5, 5],
    )
    model.eval()
    with torch.no_grad():
        for d in data:
            x = torch.stack((d["f0"], d["rms"]), dim=1)  # B, 2, frames
            prob = model(x)
            prob = torch.sigmoid(prob)
            d["pconv_trp"] = prob.cpu()
    return data


def append_pros_convQ(data):
    from turngpt.prosody_classifier import QClassifier

    checkpoint = "checkpoints/prosQClassifier/version_1/checkpoints/epoch=4-val_loss=0.47939.ckpt"
    model = QClassifier.load_from_checkpoint(
        checkpoint,
        in_channels=2,
        conv_hidden=32,
        layers=7,
        first_stride=[2, 1, 1, 1, 1, 1, 1],
        kernel=[5, 5, 5, 5, 5, 5, 5],
    )
    model.eval()
    with torch.no_grad():
        for d in data:
            x = torch.stack((d["f0"], d["rms"]), dim=1)  # B, 2, frames
            prob, *_ = model(x)
            prob = torch.sigmoid(prob)
            d["pconvq_trp"] = prob.cpu()
    return data


def append_pros_convQRNN(data):
    from turngpt.prosody_classifier import QRNNClassifier

    checkpoint = "checkpoints/prosQRNNClassifier/version_0/checkpoints/epoch=2-val_loss=0.49045.ckpt"
    model = QRNNClassifier.load_from_checkpoint(
        checkpoint,
        in_channels=2,
        conv_hidden=32,
        layers=7,
        first_stride=[2, 1, 1, 1, 1, 1, 1],
        kernel=[5, 5, 5, 5, 5, 5, 5],
    )
    model.eval()
    with torch.no_grad():
        for d in data:
            x = torch.stack((d["f0"], d["rms"]), dim=1)  # B, 2, frames
            prob, *_ = model(x)
            prob = torch.sigmoid(prob)
            d["pconvqrnn_trp"] = prob.cpu()
    return data


def append_PW(data):
    from turngpt.prosody_word_classifier import ProsodyWordClassifier

    checkpoint = "checkpoints/PW/version_1/checkpoints/epoch=18-val_loss=0.34452.ckpt"
    hparams = "checkpoints/PW/version_1/hparams.yaml"
    model = ProsodyWordClassifier.load_from_checkpoint(checkpoint, hparams_file=hparams)
    model.eval()
    with torch.no_grad():
        for d in data:
            x = torch.stack((d["f0"], d["rms"]), dim=1)  # B, 2, frames
            inp = d["input_ids"]
            prob = model(inp, x)
            prob = torch.sigmoid(prob)
            d["pw_trp"] = prob.cpu()
    return data


if __name__ == "__main__":
    # prepare()
    data = torch.load("checkpoints/phrase_data_f0n_f0i.pt")
    tokenizer = data["tokenizer"]
    data = data["data"]
    data = append_pros_conv(data)
    data = append_pros_convQ(data)
    data = append_pros_convQRNN(data)
    data = append_PW(data)
    data = append_turn_acoustic(data)
    data = append_prosody_gpt(data)
    for i, d in enumerate(data):
        tok = list(convert_ids_to_tokens(data[i]["input_ids"], tokenizer))
        data[i]["tokens"] = tok
    data.sort(key=lambda x: x["tokens"])
    _ = data.pop(0)
    _ = data.pop(2)
    sorted_data = []
    for i in range(len(data)):
        if i % 2 == 0:
            sorted_data.append({"short": data[i], "long": data[i + 1]})
    for d in sorted_data:
        print(d["short"]["tokens"])
        print(d["long"]["tokens"])
        print("-" * 50)

    # print short vs long

    key = "pconv_trp"
    key = "pconvq_trp"
    key = "pconvqrnn_trp"
    all_pros = ["pconv_trp", "pconvq_trp", "pconvqrnn_trp", "pw_trp", "trp_pros_gpt"]
    all_sequence = ["lm_trp", "trp_ta", "trp_pros_gpt"]

    plot_all_lines(sorted_data, all_sequence, all_pros, figsize=(12, 9))

    plt.close("all")

    def plot_lines(pair, keys, ax_short, ax_long, alpha=0.5):
        avg_short = []
        avg_long = []
        for k in keys:
            x_short = torch.arange(len(pair["short"][k]))
            x_long = torch.arange(len(pair["long"][k]))
            sh = pair["short"][k]
            lo = pair["long"][k]
            if k == "lm_trp":
                sh = sh.view(-1, 1)
                lo = lo.view(-1, 1)
            avg_short.append(sh)
            avg_long.append(lo)
            ax_short.plot(
                x_short,
                sh,
                label=f"short {k}",
                alpha=alpha,
            )
            ax_long.plot(
                x_long,
                lo,
                label=f"long {k}",
                alpha=alpha,
            )
            ax_short.legend(loc="upper left")
            ax_long.legend(loc="upper left")
        avg_short = torch.cat(avg_short, dim=-1).mean(dim=-1)
        avg_long = torch.cat(avg_long, dim=-1).mean(dim=-1)
        ax_short.plot(
            x_short,
            avg_short,
            color="k",
            linewidth=2,
            linestyle="dashed",
            label="short avg",
        )
        ax_long.plot(
            x_long,
            avg_long,
            color="k",
            linewidth=2,
            linestyle="dashed",
            label="long avg",
        )
        ax_short.legend(loc="upper left")
        ax_long.legend(loc="upper left")
        return ax_short, ax_long

    def plot_all_lines(sorted_data, sequence_keys, pros_keys, figsize=(12, 4)):
        width = 0.2
        offset = width
        alpha = 0.3
        fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, sharey=True)
        for pair in sorted_data:
            for a in ax:
                a.cla()
            ax[0], ax[1] = plot_lines(pair, sequence_keys, ax[0], ax[1])
            ax[2], ax[3] = plot_lines(pair, pros_keys, ax[2], ax[3])
            short_len = len(pair["short"]["tokens"]) - 1
            ax[1].vlines(short_len, 0, 1, color="k")
            ax[3].vlines(short_len, 0, 1, color="k")
            ax[1].set_ylim([0, 1])
            ax[1].set_xticks(torch.arange(len(pair["long"]["tokens"])))
            ax[1].set_xticklabels(pair["long"]["tokens"], rotation=45)
            plt.tight_layout()
            plt.pause(0.01)
            input()

    for i in range(len(data)):
        f0 = data[i]["f0"]
        ids = data[i]["input_ids"]
        tok = convert_ids_to_tokens(ids, tokenizer)
        print(" ".join(tok))
        fig, ax = plt.subplots(f0.shape[0], 1, figsize=(9, 2 * f0.shape[0]))
        for i in range(f0.shape[0]):
            ax[i].plot(f0[i], label=tok[i])
            ax[i].hlines(0, 0, len(f0[i]), linestyle="dashed", color="k", linewidth=0.5)
            ax[i].set_ylim([-3, 3])
            ax[i].legend(loc="upper left")
        plt.tight_layout()
        plt.pause(0.01)
        input()
        plt.close()

    for i in range(len(data)):
        plot_trp(
            data[i]["lm_trp"], convert_ids_to_tokens(data[i]["input_ids"], tokenizer)
        )
        input()
        plt.close()

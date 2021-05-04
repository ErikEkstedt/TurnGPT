from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import time
import umap
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from turngpt.acousticDM import AudioDM
from ttd.utils import find_island_idx_len
from ttd.tokenizer_helpers import convert_ids_to_tokens


def extract_pos_neg_samples(
    dm, split="train", features=["f0", "rms", "duration", "tokens"]
):
    if split == "train":
        loader = dm.train_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        print("Warning using TEST set")
        loader = dm.test_dataloader()
    pos = {feature: [] for feature in features}
    neg = {feature: [] for feature in features}
    # create training samples
    for batch in tqdm(loader):
        if "tokens" in features:
            batch["tokens"] = convert_ids_to_tokens(batch["input_ids"], dm.tokenizer)
        sp = batch["input_ids"] == dm.sp1_idx
        sp += batch["input_ids"] == dm.sp2_idx
        positives = sp[:, 1:]  # prior to speaker token
        # negatives: NOT prior to speaker-token or speaker token
        negatives = torch.logical_not(torch.logical_or(positives, sp[:, :-1]))
        negatives = torch.where(negatives)
        positives = torch.where(positives)
        for k in pos.keys():
            pos[k].append(batch[k][:, :-1][positives])
        for k in neg.keys():
            neg[k].append(batch[k][:, :-1][negatives])

    for k in pos.keys():
        try:
            pos[k] = torch.cat(pos[k])
        except TypeError:
            pos[k] = np.concatenate(pos[k])

    for k in neg.keys():
        try:
            neg[k] = torch.cat(neg[k])
        except TypeError:
            neg[k] = np.concatenate(neg[k])
    return pos, neg


def plot_umap(X, split, metric="cosine", label="F0", plot=False):
    t = time.time()
    projection = umap.UMAP(n_neighbors=5, min_dist=0.1, metric=metric).fit_transform(X)
    t = round(time.time() - t, 2)
    print(f"UMAP {metric}: {t} seconds")

    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        projection[:split, 0],
        projection[:split, 1],
        alpha=0.1,
        color="g",
        label=f"pos {label}",
    )
    ax.scatter(
        projection[split:, 0],
        projection[split:, 1],
        alpha=0.1,
        color="r",
        label=f"neg {label}",
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax, projection


def get_1d_derivative(x):
    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim == 2:
        x = x.unsqueeze(1)

    # Derivative
    kernel = torch.tensor([1.0, 0.0, -1.0]).view(1, 1, 3)
    der = nn.Conv1d(1, 1, kernel_size=3, bias=False)
    der.weight.data = kernel
    with torch.no_grad():
        dx = der(x).squeeze()
    return dx


if __name__ == "__main__":

    parser = ArgumentParser()
    # parser = AudioDM.add_data_specific_args(
    #     parser,
    #     datasets=["maptask"],
    #     # datasets=["switchboard"],
    #     waveform=True,
    #     f0=True,
    #     f0_normalize=True,
    #     f0_interpolate=True,
    #     f0_smooth=True,
    #     rms=True,
    #     log_rms=False,
    #     duration=True,
    # )
    parser = AudioDM.add_data_specific_args(
        parser,
        # datasets=["maptask"],
        datasets=["switchboard"],
        waveform=True,
        f0=True,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
        rms=True,
        log_rms=False,
        duration=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.word_audio_segment_time = 0.5
    args.batch_size = 16
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))

    w = batch["waveform"]

    pos["tokens"]

    pos, neg = extract_pos_neg_samples(
        dm, split="train", features=["f0", "input_ids", "waveform", "tokens"]
    )

    toks = dm.tokenizer.convert_ids_to_tokens(pos["input_ids"].tolist())

    regular = 0
    non_reg = 0
    sub = set()
    for t in toks:
        if t.startswith("Ġ"):
            regular += 1
        else:
            non_reg += 1
            sub.update([t])

    print(non_reg / (regular + non_reg))

    n_neg = neg["f0"].shape[0]
    n_pos = pos["f0"].shape[0]
    print("pos: ", n_pos)
    print("neg: ", n_neg)
    print("pos/total: ", round(n_pos / (n_pos + n_neg), 3))

    rand_neg = np.random.choice(np.arange(n_neg), n_pos, replace=False)
    F0 = torch.cat((pos["f0"], neg["f0"][rand_neg]))
    rms = torch.cat((pos["rms"], neg["rms"][rand_neg]))
    duration = torch.cat((pos["duration"], neg["duration"][rand_neg]))

    fig, ax, projection = plot_umap(F0, split=n_pos)
    plt.pause(0.01)

    dx = get_1d_derivative(F0[0])
    print("dx: ", tuple(dx.shape))

    df0 = get_1d_derivative(F0)
    d2f0 = get_1d_derivative(df0)

    fig, ax, projection = plot_umap(df0, split=n_pos, label="dF0")
    plt.pause(0.01)

    fig, ax, projection = plot_umap(d2f0, split=n_pos, label="d2F0")
    plt.pause(0.01)

    falling = df0 < 0
    rising = df0 > 0
    flat = df0 == 0

    ax.plot(torch.where(falling)[0], p[1:-1][falling], "r.")
    ax.plot(torch.where(rising)[0], p[1:-1][rising], "g.")
    ax.plot(torch.where(flat)[0], p[1:-1][flat], "k.")
    plt.pause(0.01)

    # All
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].plot(pos["f0"].T)
    ax[1].plot(neg["f0"].T)
    ax[1].set_ylim([-2.5, 2.5])
    ax[1].set_xlim([0, pos["f0"].shape[1]])
    plt.tight_layout()
    plt.pause(0.01)

    fig, ax = plt.subplots(1, 1)
    for p, tok in zip(pos["f0"], pos["tokens"]):
        ax.cla()
        df = der(p.view(1, 1, -1)).squeeze(0).squeeze(0)
        falling = df < 0
        rising = df > 0
        flat = df == 0
        ax.plot(torch.where(falling)[0], p[1:-1][falling], "r.")
        ax.plot(torch.where(rising)[0], p[1:-1][rising], "g.")
        ax.plot(torch.where(flat)[0], p[1:-1][flat], "k.")
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlim([0, len(p)])
        plt.tight_layout()
        plt.pause(0.01)
        input()

    fig, ax = plt.subplots(1, 1)
    for p, tok in zip(neg["f0"], neg["tokens"]):
        ax.cla()
        df = der(p.view(1, 1, -1)).squeeze(0).squeeze(0)
        falling = df < 0
        rising = df > 0
        flat = df == 0
        ax.plot(torch.where(falling)[0], p[1:-1][falling], "r.")
        ax.plot(torch.where(rising)[0], p[1:-1][rising], "g.")
        ax.plot(torch.where(flat)[0], p[1:-1][flat], "k.")
        ax.set_ylim([-2.5, 2.5])
        ax.set_xlim([0, len(p)])
        plt.tight_layout()
        plt.pause(0.01)
        input()

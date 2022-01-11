import matplotlib.pyplot as plt
import torch


def plot_trp(
    trp,
    proj=None,
    text=None,
    unk_token="<|endoftext|>",
    eos_token="<ts>",
    figsize=(9, 3),
    plot=True,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # omit post unk_token
    if text is not None:
        max_idx = len(text)
        for n, t in enumerate(text):
            if t == unk_token:
                max_idx = n
                break
        text = text[:max_idx]
        trp = trp[:max_idx]
        if proj is not None:
            proj = proj[:max_idx]

    x = torch.arange(len(trp))
    if proj is not None:
        ax.bar(x, proj, alpha=0.1, width=1.0, color="b", label="projection")
    ax.bar(x, trp, width=0.3, color="b", label="TRP")
    ax.set_xticks(x)
    if text is not None:
        ax.set_xticklabels(text, rotation=60)
        for i, t in enumerate(text):
            if t == eos_token:
                ax.vlines(i, ymin=0, ymax=1, linestyle="dashed", color="r", alpha=0.6)
    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


def plot_each_turn(trp, proj=None, likelihood=None, text=None):
    # split into turns
    turns = []
    start = -1
    for i in range(len(text)):
        if text[i] == "<ts>":
            tmp = {
                "trp": trp[start + 1 : i + 1],
                "text": text[start + 1 : i + 1],
            }

            if proj is not None:
                tmp["proj"] = proj[start + 1 : i + 1]

            if likelihood is not None:
                tmp["likelihood"] = likelihood[start + 1 : i + 1]

            turns.append(tmp)
            start = i

    figures = []
    for i in range(len(turns)):
        fig, ax = plt.subplots(1, 1)
        trp = turns[i]["trp"]
        text = turns[i]["text"]
        x = torch.arange(len(trp))

        if proj is not None:
            ax.bar(
                x,
                turns[i]["proj"],
                alpha=0.1,
                width=1.0,
                color="b",
                label="projection",
            )

        diff = 0
        if likelihood is not None:
            ax.bar(
                x - 0.15,
                turns[i]["likelihood"],
                width=0.3,
                color="orange",
                label="likelihood",
            )
            diff = -0.15
        ax.bar(x - diff, trp, width=0.3, color="b", label="TRP")

        ax.set_xticks(x)
        if text is not None:
            ax.set_xticklabels(text, rotation=60)
        ax.set_ylim([0, 1])
        ax.legend()
        fig.tight_layout()
        figures.append((fig, ax))
    return figures

from argparse import ArgumentParser
from turngpt.main import TurnGPT
from turngpt.turngpt_dm import TurnGPTDM
from turngpt.turngpt_utils import get_speaker_shift_indices
from ttd.tokenizer_helpers import convert_ids_to_tokens

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


def plot_turn_shift_prediction(pr, trp, tokens, sp_ind, plot=False):
    assert len(pr) == len(trp) == len(tokens)

    pr = pr.cpu()
    trp = trp.cpu()

    # rolling average
    # m = 2
    # pr_roll = torch.cat((torch.zeros(pr.shape[0], m-1), pr), dim=1).unfold(dimension=1, step=1, size=m).sum(dim=-1) / m

    # # diff
    # pr_diff = pr[:, 1:] - pr[:, :-1]
    # pr_diff[pr_diff < 0] = 0

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(pr, "r", label="proximity", alpha=0.5)
    ax.plot(trp, "b", label="trp", alpha=0.5)
    # ax.plot(pr_roll[B], "g", label="trp", alpha=0.3)
    # ax.plot(pr_diff[B], "m", label="trp", alpha=0.3)
    ax.set_xticks(torch.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=65, fontdict={"fontsize": 10})
    ax.vlines(sp_ind, ymin=0, ymax=1, color="k", alpha=0.1, linewidth=4)
    ax.hlines(
        0.5,
        xmin=0,
        xmax=len(tokens),
        color="k",
        alpha=0.2,
        linestyles="dashed",
        linewidth=0.8,
    )
    ax.set_xlim([0, len(tokens)])
    ax.legend()
    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--task", action="store_true")
    # parser = TurnGPTDM.add_data_specific_args(parser, datasets=["persona"])
    parser = TurnGPTDM.add_data_specific_args(parser)
    args = parser.parse_args()
    args.chunk_size = 128
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # args.checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained/version_0/checkpoints/epoch=2-val_loss=2.84545.ckpt"
    # args.checkpoint = "checkpoints/proxi/checkpoints/epoch=3-val_loss=2.45156.ckpt"
    model = TurnGPT.load_from_checkpoint(args.checkpoint)

    dm = TurnGPTDM(args)
    dm.prepare_data()

    if torch.cuda.is_available():
        model.cuda()

    if args.split == "val" or args.split == "train":
        dm.setup("fit")
        if args.split == "train":
            loader = dm.train_dataloader()
            print("split: Train")
        else:
            loader = dm.val_dataloader()
            print("split: Val")
    elif args.split == "test":
        dm.setup("test")
        loader = dm.test_dataloader()
        print("\nsplit: Test")
    else:  # all
        dm.setup("fit")
        dm.train_dset.filepaths += dm.val_dset.filepaths
        dm.train_dset.filepaths += dm.test_dset.filepaths
        loader = dm.train_dataloader()
        print("\nsplit: All")

    #######################################################################
    if args.task:
        trainer = pl.Trainer(gpus=1)
        out = trainer.test(model, loader, verbose=False)
        for k, v in out[0].items():
            print(f"{k}: {v}")

    if args.plot:
        import matplotlib.pyplot as plt

        for batch in loader:
            input_ids, speaker_ids = batch[0], batch[1]
            out = model(input_ids.to(model.device), speaker_ids.to(model.device))
            with torch.no_grad():
                pr = F.softmax(out["proximity_logits"], dim=-1)[..., 1]
                trp = F.softmax(out["logits"], dim=-1)
                trp = torch.stack(
                    (trp[..., model.sp1_idx], trp[..., model.sp2_idx]), dim=-1
                )
                trp, _ = trp.max(dim=-1)

            for B in range(batch[0].shape[0]):
                inp_ids = input_ids[B]
                _, sp_ind = get_speaker_shift_indices(
                    inp_ids.unsqueeze(0), model.sp1_idx, model.sp2_idx
                )
                toks = convert_ids_to_tokens(inp_ids, dm.tokenizer)
                fig, ax = plot_turn_shift_prediction(
                    pr[B], trp[B], toks, sp_ind, plot=True
                )
                try:
                    ans = input("Ctrl-C to break")
                except KeyboardInterrupt:
                    import sys

                    sys.exit(0)

from argparse import ArgumentParser
import torch
from ttd.basebuilder import add_builder_specific_args
from turngpt_dm import TurnGPTDM

import pytorch_lightning as pl

from turngpt_mini import TurnGPTMini
from turngpt_sparse import TurnGPTSparse


"""
What is going on?


"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=[
            "taskmaster",
            "metalwoz",
            "multiwoz",
            "persona",
            "dailydialog",
            "coached",
        ],
    )
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()
    args.explicit_turns = True
    args.chunk_size = 128
    dm = TurnGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    model = TurnGPTMini.load_from_checkpoint(
        checkpoint_path="turngpt_mini/runs/TurnGPTMini/version_3/checkpoints/ch_epoch=20-val_loss=3.3033.ckpt",
        tokenizer=dm.tokenizer,
    )
    model.to("cuda")

    args.gpus = 1
    trainer = pl.Trainer.from_argparse_args(args)

    # model = TurnGPTSparse.load_from_checkpoint(
    #     checkpoint_path="turngpt_mini/runs/TurnGPTSparse/version_0/checkpoints/ch_epoch=07-val_loss=3.3612.ckpt",
    #     tokenizer=dm.tokenizer,
    # )
    # model.to("cuda")

    # checkpoint_path = (
    #     "turngpt_mini/runs/TurnGPTSparse/version_0/checkpoints/ch_epoch=07-val_loss=3.3612.ckpt",
    # )

    from tqdm import tqdm

    batch = next(iter(dm.val_dataloader()))

    loss = []
    for b in tqdm(dm.val_dataloader()):
        out = model.validation_step([b[0].to("cuda"), b[1].to("cuda")])
        loss.append(out["val_loss"])

    l = torch.stack(loss).mean()

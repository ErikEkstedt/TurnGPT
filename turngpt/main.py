from argparse import ArgumentParser
from os.path import join
from os import environ

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ttd.basebuilder import add_builder_specific_args
from ttd.utils import get_run_dir

from turngpt.turngpt_dm import TurnGPTDM
from turngpt.TurnGPT import TurnGPT


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser = TurnGPT.add_model_specific_args(parser)

    # ------------------------------------------------------------------
    args = parser.parse_args()
    args.chunk_size = 128

    # ------------------------------------------------------------------
    # Data
    dm = TurnGPTDM(args)
    print("DataLoader")
    print("Batch size: ", args.batch_size)
    print("num workers: ", args.num_workers)
    print("chunk size: ", args.chunk_size)
    dm.prepare_data()
    dm.setup("fit")

    model = TurnGPT(
        n_vocab=len(dm.tokenizer),
        pad_idx=dm.tokenizer.pad_token_id,
        sp1_idx=dm.sp1_idx,
        sp2_idx=dm.sp2_idx,
        learning_rate=args.learning_rate,
        proximity_constant=args.proximity_constant,
        args=args,
    )

    print("\n-----", "Model", "-----")
    print("LM:       ", model.lm_model.__class__.__name__)
    if model.proximity_model is None:
        print("Proximity: None")
    else:
        print("Proximity: ", model.proximity_model.__class__.__name__)
    print("pad_idx: ", model.pad_idx)
    print("sp1_idx: ", model.sp1_idx)
    print("sp2_idx: ", model.sp2_idx)
    print("n_vocab: ", len(dm.tokenizer))
    print()

    # ------------------------------------------------------------------
    # Checkpoint callback (early stopping)

    # Where to save the training
    print()
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print(args.save_dir)
    print()

    checkpoint_callback = None
    callbacks = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        name = "TurnGPT" + args.model
        desc = f"{name} training"
        logger = TensorBoardLogger(args.save_dir, name=name)
        ch_path = join(logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}-{val_loss:.5f}",
            save_top_k=2,
            mode="min",
            monitor="val_loss",
        )

        # Save the used tokenizer
        tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        torch.save(dm.tokenizer, tokenizer_path)
        print("tokenizer saved -> ", tokenizer_path)

        if args.early_stopping:
            print(f"Early stopping (patience={args.patience})")
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=True,
            )
            callbacks = [early_stop_callback]
        print("-" * 50)

    # ------------------------------------------------------------------
    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Fit
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    from matplotlib import use as mpl_use

    mpl_use("Agg")

    pl.seed_everything(1234)
    main()

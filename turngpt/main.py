from argparse import ArgumentParser
from os.path import join
from os import environ

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from ttd.basebuilder import add_builder_specific_args
from ttd.utils import get_run_dir

from turngpt_dm import TurnGPTDM


def main(args):
    """ main """
    # ------------------------------------------------------------------
    # Data
    dm = TurnGPTDM(args)
    print("DataLoader")
    print("Batch size: ", args.batch_size)
    print("num workers: ", args.num_workers)

    # ------------------------------------------------------------------
    # Checkpoint callback (early stopping)
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
    # Data
    dm.prepare_data()
    dm.setup("fit")

    # ------------------------------------------------------------------
    # Model
    if args.model == "mini":
        from models.gpt_mini import TurnGPTMini

        model = TurnGPTMini(
            n_vocab=len(dm.tokenizer),
            pad_idx=dm.tokenizer.pad_token_id,
            **vars(args),
        )
    elif args.model == "pretrained":
        from models.pretrained import TurnGPTPretrained

        model = TurnGPTPretrained(
            n_vocab=len(dm.tokenizer),
            pad_idx=dm.tokenizer.pad_token_id,
            **vars(args),
        )
    elif args.model == "rnn":
        from models.gpt_rnn import TurnGPTRnn

        model = TurnGPTRnn(
            n_vocab=len(dm.tokenizer),
            pad_idx=dm.tokenizer.pad_token_id,
            **vars(args),
        )
    print("\n-----", "Model", "-----")
    print("pad_idx: ", model.pad_idx)
    print("n_vocab: ", len(dm.tokenizer))
    if "n_layer" in args:
        print("n_layer: ", args.n_layer)
    else:
        print("n_layer: ", model.n_layer)

    if "n_head" in args:
        print("n_head: ", args.n_head)
    else:
        print("n_head: ", model.n_head)
    if "n_embd" in args:
        print("n_embd: ", args.n_embd)
    else:
        print("n_embd: ", model.n_embd)
    print()

    # ------------------------------------------------------------------
    # Fit
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    from matplotlib import use as mpl_use

    mpl_use("Agg")

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser.add_argument("--early_stopping", default=False, type=bool)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument(
        "--model",
        type=str,
        default="mini",  # sparse, hugging
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["coached"],
    )

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # Choose Model
    if temp_args.model == "mini":
        from models.gpt_mini import TurnGPTMini

        parser = TurnGPTMini.add_model_specific_args(parser)
    elif temp_args.model == "sparse":
        from models.gpt_sparse import TurnGPTSparse

        parser = TurnGPTSparse.add_model_specific_args(parser)
    elif temp_args.model == "pretrained":
        from models.pretrained import TurnGPTPretrained

        parser = TurnGPTPretrained.add_model_specific_args(parser)
    elif temp_args.model == "rnn":
        from gpt_rnn import TurnGPTRnn

        parser = TurnGPTRnn.add_model_specific_args(parser)
    else:
        raise NotImplementedError(
            'The model argument must be one of "mini", "sparse" or "pretrained"'
        )

    # Add all datasets
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    # Where to save the training
    print()
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print(args.save_dir)
    print()

    main(args)

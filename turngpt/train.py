from argparse import ArgumentParser
from os import makedirs
from os.path import join
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pytorch_lightning as pl

from datasets_turntaking import ConversationalDM
from turngpt.model import TurnGPT, TurnGPTWandbCallbacks


PROJECT = "TurnGPT"
SAVE_DIR = "runs/TurnGPT"


def default_logger_callbacks(name, args, callbacks):
    makedirs(SAVE_DIR, exist_ok=True)
    logger = WandbLogger(
        save_dir=SAVE_DIR,
        project=PROJECT,
        name=name + args.name_info,
        log_model=True,
    )
    # logger.watch(model)

    id_hash = logger.experiment.path.split("/")[-1]
    ch_path = join(logger.save_dir, logger.name + "_" + id_hash)
    callbacks.append(
        ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}_{val_loss:.4f}",
            save_top_k=2,
            mode="min",
            monitor="val_loss",
        )
    )

    print(f"Early stopping (patience={args.patience})")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        strict=True,  # crash if "monitor" is not found in val metrics
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    return logger, callbacks


def train():
    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    parser = ConversationalDM.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", default=10, type=int)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Model
    print("Loading Model...")
    model = TurnGPT(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        trp_projection_steps=args.trp_projection_steps,
        trp_projection_type=args.trp_projection_type,
        weight_loss=args.weight_loss,
        weight_eos_token=args.weight_eos_token,
        weight_regular_token=args.weight_regular_token,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        pretrained=args.pretrained,
        no_train_first_n=args.no_train_first_n,
        omit_dialog_states=args.omit_dialog_states,
    )
    model.init_tokenizer()  # required for fresh model (saved on checkpoint)
    model.initialize_special_embeddings()  # required for fresh model (also performed in on_load_checkpoint)
    model.print_parameters()

    # DataModule
    dm = ConversationalDM(
        datasets=args.datasets,
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        savepath=args.savepath,
        overwrite=args.overwrite,
        load_from_cache_file=args.load_from_cache_file,
        num_proc=args.num_proc,
    )
    dm.prepare_data()

    # Callbacks & Logger
    logger = None
    callbacks = None

    # this should be handled automatically with pytorch_lightning?
    if not args.fast_dev_run:
        callbacks = [TurnGPTWandbCallbacks()]
        logger, callbacks = default_logger_callbacks(
            name=model.run_name, args=args, callbacks=callbacks
        )

    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args=args,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()

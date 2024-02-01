from os import environ, makedirs
from os.path import join

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger, WandbLogger

from turngpt.dataset import SodaDM
from turngpt.model import TurnGPT, TurnGPTWandbCallbacks

environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

PROJECT = "TurnGPT"
SAVE_DIR = "runs/TurnGPT"

mpl.use("agg")


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        stage: str,
    ) -> None:
        if isinstance(trainer.logger, Logger):
            # Required for proper reproducibility
            print("save_config")
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})


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

    environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    L.seed_everything(1)

    # Model
    print("Loading Model...")
    model = TurnGPT()
    model.init_tokenizer()  # required for fresh model (saved on checkpoint)
    model.initialize_special_embeddings()  # required for fresh model (also performed in on_load_checkpoint)
    model.print_parameters()

    # DataModule
    dm = SodaDM(
        batch_size=4,
        max_length=256,
        num_workers=4,
    )
    dm.prepare_data()

    # Callbacks & Logger
    logger = None
    callbacks = None

    # this should be handled automatically with pytorch_lightning?
    # if not args.fast_dev_run:
    #     callbacks = [TurnGPTWandbCallbacks()]
    #     logger, callbacks = default_logger_callbacks(
    #         name=model.run_name, args=args, callbacks=callbacks
    #     )

    # Trainer
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    # train()
    cli = LightningCLI(
        TurnGPT, SodaDM, run=False, save_config_callback=LoggerSaveConfigCallback
    )

    # Only for fresh models
    cli.model.init_tokenizer()  # required for fresh model (saved on checkpoint)
    cli.model.initialize_special_embeddings()  # required for fresh model (also performed in on_load_checkpoint)

    # if cli.trainer.logger is not None:
    #     print("SAVE_DIR: ", cli.trainer.logger.save_dir)
    #     for k, v in vars(cli.trainer.logger).items():
    #         print(f"{k}: {v}")
    #
    # if cli.trainer.callbacks is not None:
    #     if len(cli.trainer.callbacks) > 0:
    #         for cb in cli.trainer.callbacks:
    #             print(cb)
    #             for k, v in vars(cb).items():
    #                 print(f"{k}: {v}")
    #         print("=" * 80)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

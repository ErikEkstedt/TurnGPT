from os.path import join
from argparse import ArgumentParser
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from ttd.utils import read_json
from ttd.basebuilder import create_builders
from ttd.tokenizer import (
    load_turngpt_tokenizer,
    get_special_tokens_dict,
)


class JsonDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        x = read_json(filepath)
        return x


def collate_fn_wrapper(pad_idx=0):
    """
    A simple wrapper around the verbal_collate_fn in order to be able to provide
    arguments (e.g. pad_idx)

    RETURNS:
        verbal_collate_fn
    """

    def verbal_collate_fn(batch):
        """
        Using padding_value = -100 which by default is not used in nn.CrossEntropyLoss
        """
        sequences = []
        speaker_ids = []
        ends = []
        for b in batch:
            sequences.append(torch.tensor(b["input_ids"]))
            speaker_ids.append(torch.tensor(b["speaker_ids"]))
        return (
            pad_sequence(sequences, batch_first=True, padding_value=pad_idx).long(),
            pad_sequence(speaker_ids, batch_first=True, padding_value=pad_idx).long(),
        )

    return verbal_collate_fn


class TurnGPTDM(pl.LightningDataModule):
    """
    Wrapper around multiple dms
    """

    def __init__(self, hparams, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        self.hparams = hparams

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            special_token_dict = get_special_tokens_dict(
                hparams["tokenizer_special_dict"]
            )
            self.tokenizer = load_turngpt_tokenizer(
                pretrained=hparams["tokenizer_pretrained"],
                special_token_dict=special_token_dict,
            )

        self.pad_idx = self.tokenizer.pad_token_id
        self.builders = create_builders(hparams)

    def prepare_data(self):
        for i, builder in enumerate(self.builders):
            tok_path = builder.prepare_explicit_turn_level_tokens(
                tokenizer=self.tokenizer, EOT_token_id=self.hparams["EOT_token_id"]
            )
            if self.hparams["chunk_size"] > 0:
                tok_path = builder.prepare_chunked_tokens(
                    tok_path,
                    chunk_size=self.hparams["chunk_size"],
                    overlap=self.hparams["chunk_overlap"],
                    keep_length=self.hparams["chunk_keep_length"],
                )
                # we get more files after chunking then dialogs we defined in the splits
                # this changes the splits to include all chunks
                builder.transform_split_filepaths_with_chunks(tok_path)
            builder.task_path = tok_path  # set path for this task

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            self.train_filepaths = []
            self.val_filepaths = []

            for builder in self.builders:
                for json_name in builder.train_filepaths:
                    self.train_filepaths.append(join(builder.task_path, json_name))

                for json_name in builder.val_filepaths:
                    self.val_filepaths.append(join(builder.task_path, json_name))

            self.train_dset = JsonDataset(self.train_filepaths)
            self.val_dset = JsonDataset(self.val_filepaths)
            self.dummy_input = None

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_filepaths = []
            for builder in self.builders:
                for json_name in builder.test_filepaths:
                    self.test_filepaths.append(join(builder.task_path, json_name))

            self.test_dset = JsonDataset(self.test_filepaths)
            self.dummy_input = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--tokenizer_pretrained", type=str, default="gpt2")
        parser.add_argument(
            "--tokenizer_special_dict", type=str, default="TurnGPT_TOKENS"
        )
        parser.add_argument("--explicit_turns", type=bool, default=True)
        parser.add_argument("--EOT_token_id", type=int, default=None)
        # parser.add_argument("--chunk_size", type=int, default=-1)
        parser.add_argument("--chunk_overlap", type=int, default=10)
        parser.add_argument("--chunk_keep_length", type=int, default=20)

        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)
        return parser


if __name__ == "__main__":
    from ttd.basebuilder import add_builder_specific_args

    parser = ArgumentParser()
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["maptask", "coached"],
    )
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()
    args.explicit_turns = True
    args.chunk_size = 128
    dm = TurnGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    x = dm.train_dset[0]
    print("dset[0]: ", x.keys())

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    input_ids, speaker_ids = batch
    print("train dataloader")
    print("input_ids", input_ids.shape)
    print("speaker_ids", input_ids.shape)

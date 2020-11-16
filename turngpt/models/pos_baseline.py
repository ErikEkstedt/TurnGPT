from argparse import ArgumentParser
from os.path import join
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from ttd.utils import read_json
from ttd.basebuilder import create_builders, add_builder_specific_args
from ttd.POS import GramTrainer


class PosDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        return read_json(filepath)


# Not really a pytorch-lightning datamodule but following the format
class PosDM(object):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        self.hparams = hparams
        self.builders = create_builders(hparams)

    @staticmethod
    def add_data_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(f"--remove_punctuation", type=bool, default=True)
        return parser

    def prepare_data(self):
        for i, builder in enumerate(self.builders):
            pos_path = builder.prepare_pos()
            builder.task_path = pos_path  # set path for this task

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            self.train_filepaths = []
            self.val_filepaths = []

            for builder in self.builders:
                for json_name in builder.train_filepaths:
                    self.train_filepaths.append(join(builder.task_path, json_name))

                for json_name in builder.val_filepaths:
                    self.val_filepaths.append(join(builder.task_path, json_name))

            self.train_dset = PosDataset(self.train_filepaths)
            self.val_dset = PosDataset(self.val_filepaths)
            self.dummy_input = None

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_filepaths = []
            for builder in self.builders:
                for json_name in builder.test_filepaths:
                    self.test_filepaths.append(join(builder.task_path, json_name))

            self.test_dset = PosDataset(self.test_filepaths)
            self.dummy_input = None


def debug():
    parser = ArgumentParser()
    parser = PosDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["taskmaster", "multiwoz", "metalwoz", "coached"],
    )
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    dm = PosDM(args)
    dm.prepare_data()
    dm.setup("fit")

    pos_model = GramTrainer()
    for data in tqdm(dm.train_dset, desc=f"Train Pos"):
        pos_model.process(data["pos"])
    stats = pos_model.finalize()

    sp_uni_prob = round(100 * stats["uni"]["SPEAKER"], 2)
    print(f"Speaker: {sp_uni_prob}%")

    # bigram
    max_prob = -1
    all_sequences = []
    all_probs = []
    print("=" * 70)
    for w1, sp_prob in stats["bi"].items():
        print(f"{w1} -> {round(100*sp_prob, 2)}%")
        all_sequences.append(w1)
        all_probs.append(sp_prob)
    print("-" * 70)

    all_probs, perm_idx = torch.tensor(all_probs).sort(descending=True)
    all_seq = []
    for i in perm_idx:
        all_seq.append(all_sequences[i])
    n = 50
    for seq, prob in zip(all_seq[:n], all_probs[:n]):
        print(seq, "->", prob.item())

    # trigram
    all_sequences = []
    all_probs = []
    print("=" * 70)
    for w1, w2s in stats["tri"].items():
        for w2, sp_prob in w2s.items():
            all_sequences.append((w1, w2))
            all_probs.append(sp_prob)
    all_probs, perm_idx = torch.tensor(all_probs).sort(descending=True)
    all_seq = []
    for i in perm_idx:
        all_seq.append(all_sequences[i])
    n = 50
    for seq, prob in zip(all_seq[:n], all_probs[:n]):
        print(seq, prob.item())


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = PosDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["taskmaster", "multiwoz", "metalwoz", "coached"],
    )
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    dm = PosDM(args)
    dm.prepare_data()
    dm.setup("fit")

    pos_model = GramTrainer()
    for data in tqdm(dm.train_dset, desc=f"Train Pos"):
        pos_model.process(data["pos"])
    stats = pos_model.finalize()

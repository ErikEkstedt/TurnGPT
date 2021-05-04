from argparse import ArgumentParser
from os.path import join, exists
from os import makedirs
from librosa import time_to_frames
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import pytorch_lightning as pl

from turngpt.F0 import f0_z_normalize, interpolate_forward
from turngpt.transforms import Gaussian1D
from ttd.utils import read_json
from ttd.basebuilder import create_builders, add_builder_specific_args

from typing import List


def add_f0_args(
    parent_parser,
    f0=True,
    f0_normalize=False,
    f0_interpolate=False,
    f0_smooth=False,
    f0_mask=True,
):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--f0", action="store_true", default=f0)
    parser.add_argument("--f0_normalize", action="store_true", default=f0_normalize)
    parser.add_argument("--f0_interpolate", action="store_true", default=f0_interpolate)
    parser.add_argument("--f0_smooth", action="store_true", default=f0_smooth)
    parser.add_argument("--f0_min", type=int, default=60)
    parser.add_argument("--f0_max", type=int, default=300)
    parser.add_argument("--f0_threshold", type=float, default=0.3)
    parser.add_argument("--f0_vad_mask", action="store_true", default=f0_mask)
    return parser


class ProsClassifyDataset(Dataset):
    def __init__(
        self,
        positives=None,
        negatives=None,
        negative_sample_strategy="closest",
        load_path=None,
    ):

        if load_path is not None:
            self.samples, self.labels = torch.load(load_path)
        else:
            self.positives = (
                positives  # torch.tensor (n_positives, 2, n_frames)  voiced and f0
            )
            self.positive_labels = torch.ones(len(self.positives))  # (n_positives)
            self.n_frames = self.positives.shape[-1]

            self.negatives = negatives  # list of (2, varying_sizes)  voiced and f0
            self.negative_sample_strategy = negative_sample_strategy

            # samples one negative for each positive
            self.sampled_negatives = self.sample_negatives()

            # add extra negatives to be balanced
            self.fill_negative_samples()

            # add labels
            self.sampled_negatives_labels = torch.zeros(len(self.sampled_negatives))

            self.samples = torch.cat(
                (self.positives, self.sampled_negatives)
            )  # (n_samples, n_frames)
            self.labels = torch.cat(
                (self.positive_labels, self.sampled_negatives_labels)
            ).unsqueeze(
                -1
            )  # (n_samples, 1)

    def fill_negative_samples(self):
        i = 0
        print("Sampling more negatives for balance...")
        while len(self.sampled_negatives) < len(self.positives):
            idx = torch.randint(0, len(self.negatives), size=(1,))
            neg = self.negatives[idx]
            while neg.shape[-1] < self.n_frames + 0.5:
                idx = torch.randint(0, len(self.negatives), size=(1,))
                neg = self.negatives[idx]
            self.sampled_negatives = torch.cat(
                (self.sampled_negatives, self.sample_random(neg).unsqueeze(0))
            )
            i += 1
        print(f"added {i} extra negative samples")

    def sample_negatives(self):
        if self.negative_sample_strategy == "closest":
            sample_func = self.sample_closest
        elif self.negative_sample_strategy == "random":
            sample_func = self.sample_random
        else:
            raise NotImplementedError(
                f'strategy {self.negative_sample_strategy} not implemented.\nTry ["closest", "random"]'
            )
        sampled_negatives = []
        for n in self.negatives:
            sampled_negatives.append(sample_func(n))
        return torch.stack(sampled_negatives)

    def sample_closest(self, negative):
        n = negative[:, -self.n_frames :]
        return n

    def sample_random(self, negative):
        start = torch.randint(
            low=0, high=negative.shape[-1] - self.n_frames + 1, size=(1,)
        )
        return negative[:, start : start + self.n_frames]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = self.labels[idx]
        return {"x": x, "y": y}


class ProsodyClassificationDM(pl.LightningDataModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        self.hparams = hparams

        # audio
        self.sr = hparams["sample_rate"]
        self.hop_time = hparams["hop_time"]
        self.hop_length = int(self.sr * self.hop_time)

        # prosody samples
        self.segment_time = hparams["segment_time"]
        self.buffer_time = hparams["buffer_time"]
        self.segment_frames = int(self.segment_time / self.hop_time)
        self.negative_sample_strategy = hparams["negative_sample_strategy"]

        # Features
        self.waveform = hparams["waveform"]
        self.f0 = hparams["f0"]
        self.f0_min = hparams["f0_min"]
        self.f0_max = hparams["f0_max"]
        self.f0_threshold = hparams["f0_threshold"]
        self.f0_vad_mask = hparams["f0_vad_mask"]
        self.f0_interpolate = hparams["f0_interpolate"]
        self.f0_normalize = hparams["f0_normalize"]
        self.f0_smooth = hparams["f0_smooth"]

        # Training
        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_workers"]

        # builder
        self.builders = create_builders(hparams)

    def get_data_path(self, builder):
        return join(builder.root, "prosody_classifier")

    def prepare_data(self):
        for i, builder in enumerate(self.builders):
            builder.prepare_turn_level()  # prepare turns from continuous dialog
            f0_path = builder.prepare_f0(
                sr=self.sr,
                hop_time=self.hop_time,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                f0_threshold=self.f0_threshold,
                vad_mask=self.f0_vad_mask,
            )
            builder.f0_path = f0_path

            for builder in self.builders:
                savepath = self.get_data_path(builder)
                makedirs(savepath, exist_ok=True)
                train_data_path = join(savepath, "train_data.pt")
                if not exists(train_data_path):
                    train_filepaths = []
                    for json_name in builder.train_filepaths:
                        text_path = join(builder.turn_level_root, json_name)
                        f0_path = join(
                            builder.f0_path, json_name.replace(".json", ".pt")
                        )
                        train_filepaths.append((text_path, f0_path))

                    pos, neg = self.extract_dataset(train_filepaths, "train")
                    torch.save([pos, neg], train_data_path)

                val_data_path = join(savepath, "val_data.pt")
                if not exists(val_data_path):
                    val_filepaths = []
                    for json_name in builder.val_filepaths:
                        text_path = join(builder.turn_level_root, json_name)
                        f0_path = join(
                            builder.f0_path, json_name.replace(".json", ".pt")
                        )
                        val_filepaths.append((text_path, f0_path))

                    pos, neg = self.extract_dataset(val_filepaths, "val")
                    torch.save([pos, neg], val_data_path)
                    print(f"save {builder.NAME} val data")

                test_data_path = join(savepath, "test_data.pt")
                if not exists(test_data_path):
                    test_filepaths = []
                    for json_name in builder.test_filepaths:
                        text_path = join(builder.turn_level_root, json_name)
                        f0_path = join(
                            builder.f0_path, json_name.replace(".json", ".pt")
                        )
                        test_filepaths.append((text_path, f0_path))
                    pos, neg = self.extract_dataset(test_filepaths, "test")
                    torch.save([pos, neg], test_data_path)
                    print(f"save {builder.NAME} test data")

    def extract_dataset(self, filepaths, split):
        positives = []
        negatives = []
        for tpath, f0path in tqdm(filepaths, desc=f"process {split}"):
            try:
                turns = read_json(tpath)
                f0_dict = torch.load(f0path)
            except FileNotFoundError:
                print(f"{tpath} not found... skipping.")
                continue

            f0_dict["voiced"] = f0_dict["f0"] > 0

            if self.f0_normalize:
                f0_dict["f0"] = f0_z_normalize(
                    f0_dict["f0"], mean=f0_dict["mean"], std=f0_dict["std"]
                )

            if self.f0_interpolate:
                f0_dict["f0"] = interpolate_forward(f0_dict["f0"], f0_dict["voiced"])

            if self.f0_smooth:
                f0_dict["f0"] = Gaussian1D(kernel_size=5)(
                    f0_dict["f0"].unsqueeze(-2)
                ).squeeze(-2)

            for t in turns:
                channel = t["speaker_id"]
                assert (
                    channel == 0 or channel == 1
                ), "Error: channel must be either 0 or 1."

                s, e = t["starts"][0], t["ends"][-1]
                turn_duration = e - s

                if turn_duration > self.segment_time:
                    fe = torch.tensor(
                        time_to_frames(e, sr=self.sr, hop_length=self.hop_length)
                    )
                    f0 = f0_dict["f0"][channel, fe - self.segment_frames : fe]
                    voiced = f0_dict["voiced"][channel, fe - self.segment_frames : fe]
                    positives.append(torch.stack((f0, voiced)))

                if turn_duration > (2 * self.segment_time + self.buffer_time):
                    neg_end = e - self.segment_time - self.buffer_time
                    fs, fe = torch.tensor(
                        time_to_frames(
                            (s, neg_end), sr=self.sr, hop_length=self.hop_length
                        )
                    )
                    f0 = f0_dict["f0"][channel, fs:fe]
                    voiced = f0_dict["voiced"][channel, fs:fe]
                    negatives.append(torch.stack((f0, voiced)))

        return torch.stack(positives), negatives

    def setup(self, stage="fit"):
        if stage == "fit":
            train_positives = []
            train_negatives = []
            val_positives = []
            val_negatives = []
            for builder in self.builders:
                savepath = self.get_data_path(builder)
                pos, neg = torch.load(join(savepath, "train_data.pt"))
                train_positives.append(pos)
                train_negatives += neg  # make list larger
                pos, neg = torch.load(join(savepath, "val_data.pt"))
                val_positives.append(pos)
                val_negatives += neg  # make list larger
            train_positives = torch.cat(train_positives)
            val_positives = torch.cat(val_positives)
            self.train_dset = ProsClassifyDataset(
                train_positives, train_negatives, self.negative_sample_strategy
            )
            self.val_dset = ProsClassifyDataset(
                val_positives, val_negatives, self.negative_sample_strategy
            )

        if stage == "test":
            test_positives = []
            test_negatives = []
            for builder in self.builders:
                savepath = self.get_data_path(builder)
                pos, neg = torch.load(join(savepath, "test_data.pt"))
                test_positives.append(pos)
                test_negatives += neg  # make list larger
            test_positives = torch.cat(test_positives)
            self.test_dset = ProsClassifyDataset(
                test_positives, test_negatives, self.negative_sample_strategy
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    @staticmethod
    def add_data_specific_args(
        parent_parser,
        datasets=["maptask"],
        waveform=False,
        f0=True,
        f0_normalize=False,
        f0_interpolate=False,
        f0_smooth=False,
        f0_mask=True,
    ):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Training
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)

        # Features
        parser.add_argument("--waveform", action="store_true", default=waveform)
        parser = add_f0_args(
            parser,
            f0=f0,
            f0_normalize=f0_normalize,
            f0_interpolate=f0_interpolate,
            f0_smooth=f0_smooth,
            f0_mask=f0_mask,
        )

        parser.add_argument("--segment_time", type=float, default=1)
        parser.add_argument("--negative_sample_strategy", type=str, default="random")
        parser.add_argument("--buffer_time", type=float, default=0.5)

        # Audio params
        parser.add_argument("--sample_rate", type=int, default=8000)
        parser.add_argument("--hop_time", type=float, default=0.01)

        # datasets
        parser.add_argument(
            "--datasets",
            nargs="*",
            type=str,
            default=datasets,
        )

        temp_args, _ = parser.parse_known_args()
        parser = add_builder_specific_args(
            parser, temp_args.datasets
        )  # add for all builders
        return parser


if __name__ == "__main__":

    # maptask q6ec2 Error
    parser = ArgumentParser()
    parser = ProsodyClassificationDM.add_data_specific_args(
        parser,
        # datasets=["switchboard"],
        datasets=["maptask"],
    )
    args = parser.parse_args()
    args.batch_size = 256
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    dm = ProsodyClassificationDM(args)
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    pos = 0
    total = 0
    for batch in dm.train_dataloader():
        pos += batch["y"].sum()
        total += len(batch["y"])
    print(pos / total)

    d = dm.train_dset[0]
    print("d['x']: ", tuple(d["x"].shape))
    print("d['y']: ", tuple(d["y"].shape))

    mini = 999999
    for neg in dm.train_neg:
        n = neg.shape[-1]
        if n < mini:
            mini = n

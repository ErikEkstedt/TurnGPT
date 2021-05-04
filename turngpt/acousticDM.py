from argparse import ArgumentParser
from os.path import join, basename, exists, split
from os import makedirs, listdir
from glob import glob
from tqdm import tqdm
import math
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as AT
import pytorch_lightning as pl

from ttd.basebuilder import create_builders, add_builder_specific_args
from ttd.utils import read_json, write_json
from ttd.tokenizer import (
    load_turngpt_tokenizer,
    get_special_tokens_dict,
)

from turngpt.F0 import f0_z_normalize, interpolate_forward


# def get_f0_path(root, sr, hop_time):
#     f0_path = join(root, "f0")
#     f0_path += f"_sr-{sr}_hop-{hop_time}"
#     return f0_path

# This is defined in basebuilder and is a temporary hack
def get_f0_path(root, sr, hop_time, f0_min, f0_max, f0_threshold, vad_mask):
    fpath = f"F0_sr-{sr}_ht-{hop_time}_fmin-{f0_min}_fmax-{f0_max}_thr-{f0_threshold}_mask-{vad_mask}"
    return join(root, fpath)


def get_session_name(name):
    return name.replace(".json", "").split("_#")[0]


class Gaussian1D(nn.Module):
    """
    Inspiration:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/2
    """

    def __init__(self, channels=1, kernel_size=5, sigma=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.gaussian_filter = self._gaussian_kernel(
            self.kernel_size, self.sigma, self.channels
        )

    def _gaussian_kernel(self, kernel_size=3, sigma=2, channels=1):
        """
        From:
            https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        """

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size).unsqueeze(-1)
        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((x_coord - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # reshape for nn.Conv1d
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
        # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)

        gaussian_filter = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter

    def forward(self, x):
        return self.gaussian_filter(x)


class AcousticDataset(Dataset):
    def __init__(
        self,
        filepaths,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        waveform=False,
        rms=False,
        log_rms=False,
        duration=False,
        pre_silence=False,
        f0=False,
        f0_min=60,
        f0_max=300,
        f0_threshold=0.3,
        f0_vad_mask=True,
        f0_normalize=False,
        f0_interpolate=False,
        f0_smooth=False,
        sr=8000,
        hop_time=0.01,
        word_audio_segment_time=1.0,
        chunk_size=128,
    ):
        self.filepaths = filepaths
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.chunk_size = chunk_size

        # features
        self.rms = rms
        self.duration = duration
        if rms:
            waveform = True
        self.log_rms = log_rms
        self.waveform = waveform
        self.pre_silence = pre_silence

        self.f0 = f0
        self.f0_normalize = f0_normalize
        self.f0_interpolate = f0_interpolate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_threshold = f0_threshold
        self.f0_vad_mask = f0_vad_mask
        self.f0_smooth = f0_smooth
        if f0_smooth:
            self.f0_smoother = Gaussian1D(kernel_size=5)

        # audio parameters
        self.sr = sr
        self.hop_time = hop_time
        self.hop_length = int(sr * hop_time)
        self.word_audio_segment_time = word_audio_segment_time

        self.n_samples = int(sr * word_audio_segment_time)
        self.n_frames = self.n_samples // self.hop_length

    def __len__(self):
        return len(self.filepaths)

    def get_root(self, path):
        return split(split(path)[0])[0]

    def get_pre_silence(self, starts, ends):
        # pre_silence = torch.stack((torch.tensor(starts), torch.tensor(ends)), dim=-1)
        pre_silence = torch.tensor(starts)[1:] - torch.tensor(ends)[:-1]
        # pre_silence[pre_silence < 0] = 0
        return pre_silence

    def extract_waveform_segments(self, input_ids, channel, starts, ends, audio):
        offset = int(min(starts) * self.sr)
        ends = (
            torch.tensor(ends) * self.sr
        ).round().long() - offset  # chunked audio requires offset

        waveform, tmp_sr = torchaudio.load(audio)
        if tmp_sr != self.sr:
            resampler = AT.Resample(orig_freq=tmp_sr, new_freq=self.sr)
            waveform = resampler(waveform)

        waveform_stack = []
        for ch, ids, end_sample in zip(channel, input_ids, ends):
            if ids == self.sp1_idx or ids == self.sp2_idx:
                wav = torch.zeros(self.n_samples)
            else:
                start_sample = max(end_sample - self.n_samples, 0)
                wav = waveform[ch, start_sample:end_sample]
                if len(wav) != self.n_samples:
                    wav = torch.cat((torch.zeros(self.n_samples - len(wav)), wav))
            waveform_stack.append(wav)

        return torch.stack(waveform_stack)

    def extract_rms(self, waveform):
        """
        librosa rms
        """
        # waveform: B, n_samples
        rms = []
        for w in waveform:
            _rms = torch.from_numpy(
                librosa.feature.rms(
                    w, hop_length=self.hop_length, frame_length=2 * self.hop_length
                )[0, :-1]
            )  # omit last frame for size
            rms.append(_rms)
        return torch.stack(rms)

    def extract_duration(self, starts, ends):
        return 1 / (torch.tensor(ends) - torch.tensor(starts))

    def load_f0(self, root, name):
        f0_root = get_f0_path(
            root,
            self.sr,
            self.hop_time,
            self.f0_min,
            self.f0_max,
            self.f0_threshold,
            self.f0_vad_mask,
        )
        return torch.load(join(f0_root, name + ".pt"))

    def extract_f0_segments(self, input_ids, channel, ends, f0):
        frame_ends = torch.tensor(
            librosa.time_to_frames(ends, sr=self.sr, hop_length=self.hop_length)
        )
        frame_starts = frame_ends - self.n_frames
        frame_starts[frame_starts < 0] = 0

        f0_stack = []
        for ch, ids, start, end in zip(channel, input_ids, frame_starts, frame_ends):
            if ids == self.sp1_idx or ids == self.sp2_idx:
                tmp_f0 = torch.zeros(self.n_frames)
            else:
                tmp_f0 = f0[ch, start:end]
                if len(tmp_f0) != self.n_frames:  # beginning of audio
                    tmp_f0 = torch.cat(
                        (
                            torch.ones(self.n_frames - len(tmp_f0)) * tmp_f0[0],
                            tmp_f0,
                        )  # pad left
                    )
            f0_stack.append(tmp_f0)
        return torch.stack(f0_stack)

    def pad_to_chunk_size(self, ret):
        for k, v in ret.items():
            diff = self.chunk_size - len(v)
            if diff > 0:
                if k in ["input_ids", "speaker_ids"]:
                    ret[k] = torch.cat((v, torch.tensor([self.pad_idx] * diff)))
                else:
                    ret[k] = torch.cat((v, torch.zeros((diff, v.shape[-1]))))
        return ret

    def __getitem__(self, idx):
        text_data_path = self.filepaths[idx]
        data = read_json(text_data_path)
        speaker_ids = torch.tensor(data["speaker_ids"])
        input_ids = torch.tensor(data["input_ids"])

        ret = {"input_ids": input_ids, "speaker_ids": speaker_ids}
        channel = (speaker_ids == self.sp2_idx).long()  # relevant speaker

        if self.waveform:
            waveform_stack = self.extract_waveform_segments(
                input_ids, channel, data["starts"], data["ends"], data["audio"]
            )
            assert (
                len(input_ids) == len(speaker_ids) == len(waveform_stack)
            ), "waveform data not same length"
            ret["waveform"] = waveform_stack

        # load f0 if needed
        if self.f0:
            root = self.get_root(text_data_path)
            name = get_session_name(basename(text_data_path))
            f0_dict = self.load_f0(root, name)
            f0_dict["voiced"] = f0_dict["f0"] > 0

            if self.f0_normalize:
                f0_dict["f0"] = f0_z_normalize(
                    f0_dict["f0"], mean=f0_dict["mean"], std=f0_dict["std"]
                )

            if self.f0_interpolate:
                f0_dict["f0"] = interpolate_forward(f0_dict["f0"], f0_dict["voiced"])

            if self.f0_smooth:
                f0_dict["f0"] = self.f0_smoother(f0_dict["f0"].unsqueeze(-2)).squeeze(
                    -2
                )

            try:
                f0 = self.extract_f0_segments(
                    input_ids, channel, data["ends"], f0_dict["f0"]
                )
            except:
                print("f0: ", f0_dict["f0"].shape)
                print("path: ", text_data_path)
                print(data["starts"][0], data["ends"][-1])
            assert len(f0) == len(
                input_ids
            ), f"f0 data not same length {f0.shape}, {input_ids.shape}"
            ret["f0"] = f0

        if self.rms:
            ret["rms"] = self.extract_rms(ret["waveform"])
            if self.log_rms:
                ret["rms"] = torch.log(ret["rms"] + 1e-8)
            assert len(ret["rms"]) == len(
                input_ids
            ), f"rms not same length {ret['rms'].shape}, {input_ids.shape}"

        if self.duration:
            ret["duration"] = self.extract_duration(data["starts"], data["ends"])

        if self.pre_silence:
            ps = self.get_pre_silence(data["starts"], data["ends"])
            assert len(ps) == len(
                input_ids
            ), f"pre-silence not same length {ps.shape}, {input_ids.shape}"
            ret["pre_silence"] = ps

        if self.chunk_size > 0:
            ret = self.pad_to_chunk_size(ret)

        return ret


class AudioDM(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        self.hparams = hparams

        # Word Params & Tokenizer
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
        self.sp1_idx = self.tokenizer.convert_tokens_to_ids("<speaker1>")
        self.sp2_idx = self.tokenizer.convert_tokens_to_ids("<speaker2>")
        self.pad_idx = self.tokenizer.pad_token_id
        self.chunk_size = hparams["chunk_size"]

        # audio
        self.sr = hparams["sample_rate"]
        self.hop_time = hparams["hop_time"]
        self.hop_length = int(self.sr * self.hop_time)
        self.word_audio_segment_time = hparams["word_audio_segment_time"]
        self.pre_silence = hparams["pre_silence"]
        self.word_window_size = int(self.sr * self.word_audio_segment_time)

        # Features
        self.waveform = hparams["waveform"]
        self.rms = hparams["rms"]
        self.log_rms = hparams["log_rms"]
        self.duration = hparams["duration"]

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

    def prepare_chunked_waveform(self, builder, chunked_path):
        """
        Extracting audio segments correlated with the relevant chunk
        """
        audio_path = join(chunked_path, "audio")

        # Prepare audio for chunked-tokenized dialogs
        # audio might be slow to load because some dialogs are long
        # new_chunked_path = chunked_path + "_" + "audio"
        if not exists(audio_path) or len(listdir(audio_path)) < 10:
            makedirs(audio_path, exist_ok=True)
            for text_data_path in tqdm(
                glob(join(chunked_path, "*.json")),
                desc=f"Chunk audio {builder.NAME}",
            ):
                text_data = read_json(text_data_path)
                chunk_name = basename(text_data_path).replace(".json", "")
                name = chunk_name.split("_#")[0]

                # load audio
                waveform, tmp_sr = torchaudio.load(builder.get_audio_path(name))
                if tmp_sr != self.sr:
                    resampler = AT.Resample(orig_freq=tmp_sr, new_freq=self.sr)
                    waveform = resampler(waveform)

                start = int(min(text_data["starts"]) * self.sr)
                end = round(max(text_data["ends"]) * self.sr)
                tmp_waveform = waveform[:, start:end]

                try:
                    # save audio clip for this chunked segment
                    tmp_audio_path = join(audio_path, f"{chunk_name}_audio.wav")
                    torchaudio.save(tmp_audio_path, tmp_waveform, sample_rate=self.sr)
                    text_data["audio"] = tmp_audio_path

                    # overwrite the data with the added audio path
                    write_json(text_data, text_data_path)
                except:
                    print("Broken: ", chunk_name)

    def prepare_data(self):
        for i, builder in enumerate(self.builders):
            # prepare explicit turns
            tok_path = builder.prepare_explicit_turn_level_tokens(
                tokenizer=self.tokenizer, EOT_token_id=self.hparams["EOT_token_id"]
            )

            # preprocess F0 for each audio file
            if self.f0:
                f0_path = builder.prepare_f0(
                    sr=self.sr,
                    hop_time=self.hop_time,
                    f0_min=self.f0_min,
                    f0_max=self.f0_max,
                    f0_threshold=self.f0_threshold,
                    vad_mask=self.f0_vad_mask,
                )

            # prepare chunks
            chunked_path = builder.prepare_chunked_tokens(
                tok_path,
                chunk_size=self.hparams["chunk_size"],
                overlap=self.hparams["chunk_overlap"],
                keep_length=self.hparams["chunk_keep_length"],
            )

            # splits audio for tokenized chunks for faster load
            self.prepare_chunked_waveform(builder, chunked_path)

            # we get more files after chunking then dialogs we defined in the splits
            # this changes the splits to include all chunks
            builder.task_path = chunked_path  # used in dm.setup()
            builder.transform_split_filepaths_with_chunks(builder.task_path)

    def _dataset(self, filepaths):
        return AcousticDataset(
            filepaths,
            sp1_idx=self.sp1_idx,
            sp2_idx=self.sp2_idx,
            pad_idx=self.pad_idx,
            sr=self.sr,
            hop_time=self.hop_time,
            word_audio_segment_time=self.word_audio_segment_time,
            chunk_size=self.hparams["chunk_size"],
            waveform=self.waveform,
            rms=self.rms,
            log_rms=self.log_rms,
            pre_silence=self.pre_silence,
            duration=self.duration,
            f0=self.f0,
            f0_normalize=self.f0_normalize,
            f0_interpolate=self.f0_interpolate,
            f0_smooth=self.f0_smooth,
            f0_threshold=self.f0_threshold,
            f0_vad_mask=self.f0_vad_mask,
        )

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            self.train_filepaths = []
            self.val_filepaths = []

            for builder in self.builders:
                for json_name in builder.train_filepaths:
                    text_path = join(builder.task_path, json_name)
                    self.train_filepaths.append(text_path)

                for json_name in builder.val_filepaths:
                    text_path = join(builder.task_path, json_name)
                    self.val_filepaths.append(text_path)

            self.train_dset = self._dataset(self.train_filepaths)
            self.val_dset = self._dataset(self.val_filepaths)

        if stage == "test":
            self.test_filepaths = []
            for builder in self.builders:
                for json_name in builder.test_filepaths:
                    text_path = join(builder.task_path, json_name)
                    self.test_filepaths.append(text_path)

            self.test_dset = self._dataset(self.test_filepaths)

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
    def add_f0_args(
        parent_parser,
        f0=False,
        f0_normalize=False,
        f0_interpolate=False,
        f0_smooth=False,
        f0_mask=True,
    ):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--f0", action="store_true", default=f0)
        parser.add_argument("--f0_normalize", action="store_true", default=f0_normalize)
        parser.add_argument(
            "--f0_interpolate", action="store_true", default=f0_interpolate
        )
        parser.add_argument("--f0_smooth", action="store_true", default=f0_smooth)
        parser.add_argument("--f0_min", type=int, default=60)
        parser.add_argument("--f0_max", type=int, default=300)
        parser.add_argument("--f0_threshold", type=float, default=0.3)
        parser.add_argument("--f0_vad_mask", action="store_true", default=f0_mask)
        return parser

    @staticmethod
    def add_data_specific_args(
        parent_parser,
        datasets=["maptask"],
        waveform=False,
        rms=False,
        log_rms=False,
        pre_silence=False,
        duration=False,
        f0=False,
        f0_normalize=False,
        f0_interpolate=False,
        f0_smooth=False,
        f0_mask=True,
    ):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Tokens
        parser.add_argument("--tokenizer_pretrained", type=str, default="gpt2")
        parser.add_argument(
            "--tokenizer_special_dict", type=str, default="TurnGPT_TOKENS"
        )
        parser.add_argument("--explicit_turns", type=bool, default=True)
        parser.add_argument("--EOT_token_id", type=int, default=None)
        parser.add_argument("--chunk_size", type=int, default=128)
        parser.add_argument("--chunk_overlap", type=int, default=10)
        parser.add_argument("--chunk_keep_length", type=int, default=20)

        # Training
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)

        # Features
        parser.add_argument("--waveform", action="store_true", default=waveform)
        parser.add_argument("--rms", action="store_true", default=rms)
        parser.add_argument("--log_rms", action="store_true", default=log_rms)
        parser.add_argument("--duration", action="store_true", default=duration)
        parser.add_argument("--pre_silence", action="store_true", default=pre_silence)
        parser = AudioDM.add_f0_args(
            parser,
            f0=f0,
            f0_normalize=f0_normalize,
            f0_interpolate=f0_interpolate,
            f0_smooth=f0_smooth,
            f0_mask=f0_mask,
        )

        # Audio params
        parser.add_argument("--word_audio_segment_time", type=float, default=1)
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


def main():
    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=True,
        normalize_f0=True,
        interpolate_f0=True,
        rms=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    print("Check dataloaders are functional")
    loader = dm.train_dataloader()
    for batch in tqdm(loader, desc="train"):
        pass

    loader = dm.val_dataloader()
    for batch in tqdm(loader, desc="val"):
        pass

    dm.setup("test")
    loader = dm.test_dataloader()
    for batch in tqdm(loader, desc="test"):
        pass


if __name__ == "__main__":

    pl.seed_everything(1234)
    import sounddevice as sd
    from ttd.tokenizer_helpers import convert_ids_to_tokens
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        # datasets=["maptask"],
        datasets=["switchboard"],
        f0=True,
        waveform=True,
        normalize_f0=True,
        interpolate_f0=True,
        rms=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.duration = True
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    d = dm.train_dset[0]
    print(d.keys())

    tokens = convert_ids_to_tokens(d["input_ids"], dm.tokenizer)
    fig, ax = plt.subplots(1, 1)
    for f, t, t1 in zip(d["f0"][:-1], tokens[:-1], tokens[1:]):
        ax.cla()
        ax.set_title(t + " -> " + t1)
        ax.set_ylim([-3, 3])
        ax.plot(f)
        plt.pause(0.01)
        input()

from argparse import ArgumentParser
from os.path import join, basename, exists, split
from os import makedirs, listdir
from glob import glob
from tqdm import tqdm
import time
import librosa

import torch
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

from turngpt.F0 import F0, f0_z_normalize, interpolate_forward, F0_swipe


def get_f0_path(root, sr, hop_time):
    f0_path = join(root, "f0")
    f0_path += f"_sr-{sr}_hop-{hop_time}"
    return f0_path


def get_session_name(name):
    return name.replace(".json", "").split("_#")[0]


class AcousticDataset(Dataset):
    def __init__(
        self,
        filepaths,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        f0=False,
        rms=False,
        duration=False,
        waveform=False,
        post_silence=False,
        sr=8000,
        hop_time=0.01,
        word_audio_segment_time=1.0,
        chunk_size=128,
        normalize_f0=False,
        interpolate_f0=False,
        log_rms=False,
    ):
        self.filepaths = filepaths
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.chunk_size = chunk_size

        # features
        self.f0 = f0
        self.rms = rms
        self.duration = duration
        if rms:
            waveform = True
        self.normalize_f0 = normalize_f0
        self.interpolate_f0 = interpolate_f0
        self.log_rms = log_rms
        self.waveform = waveform
        self.post_silence = post_silence

        # audio parameters
        self.sr = sr
        self.hop_time = hop_time
        self.hop_length = int(sr * hop_time)
        self.word_audio_segment_time = word_audio_segment_time

        self.n_samples = int(sr * word_audio_segment_time)
        self.n_frames = self.n_samples // self.hop_length

    def __len__(self):
        return len(self.filepaths)

    def get_post_silence(self, starts, ends):
        post_silence = torch.stack((torch.tensor(starts), torch.tensor(ends)), dim=-1)
        # post_silence = torch.tensor(starts)[1:] - torch.tensor(ends)[:-1]
        # post_silence[post_silence < 0] = 0
        return post_silence

    def load_f0(self, root, name):
        f0_root = get_f0_path(root, self.sr, self.hop_time)
        f0_dict = torch.load(join(f0_root, name + ".pt"))
        assert self.sr == f0_dict["sr"], "sr must match sr in f0"

        hop_time = f0_dict["hop_time"]
        assert (
            self.hop_time == hop_time
        ), f"hop_length must match hop_length in prosody, {hop_time}!={self.hop_time}"
        return f0_dict

    def get_root(self, path):
        return split(split(path)[0])[0]

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

    def extract_rms(self, waveform):
        # waveform: B, n_samples
        rms = []
        for w in waveform:
            _rms = torch.from_numpy(
                librosa.feature.rms(w, hop_length=80, frame_length=160)[0, :-1]
            )  # omit last frame for size
            rms.append(_rms)
        return torch.stack(rms)

    def extract_duration(self, starts, ends):
        return 1 / (torch.tensor(ends) - torch.tensor(starts))

    def pad_to_chunk_size(self, ret):
        # if len(ret['input_ids']) != self.chunk_size:
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

            if self.normalize_f0:
                f0_dict["f0"] = f0_z_normalize(
                    f0_dict["f0"], mean=f0_dict["mean"], std=f0_dict["std"]
                )

            if self.interpolate_f0:
                f0_dict["f0"] = interpolate_forward(f0_dict["f0"], f0_dict["voiced"])

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

        if self.post_silence:
            ps = self.get_post_silence(data["starts"], data["ends"])
            assert len(ps) == len(
                input_ids
            ), f"post-silence not same length {ps.shape}, {input_ids.shape}"
            ret["post_silence"] = ps

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

        # Features
        self.waveform = hparams["waveform"]
        self.f0 = hparams["f0"]
        self.rms = hparams["rms"]
        self.log_rms = hparams["log_rms"]
        self.interpolate_f0 = hparams["interpolate_f0"]
        self.normalize_f0 = hparams["normalize_f0"]

        # audio
        self.sr = hparams["sample_rate"]
        self.hop_time = hparams["hop_time"]
        self.hop_length = int(self.sr * self.hop_time)
        self.word_audio_segment_time = hparams["word_audio_segment_time"]
        self.post_silence = hparams["post_silence"]
        self.word_window_size = int(self.sr * self.word_audio_segment_time)

        # Training
        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_workers"]

        # builder
        self.builders = create_builders(hparams)

    def preprocess_f0(self, builder, tok_path):
        f0_path = get_f0_path(builder.root, self.sr, self.hop_time)

        if not exists(f0_path) or len(listdir(f0_path)) < 10:
            makedirs(f0_path, exist_ok=True)
            _F0 = F0(sr=self.sr, hop_time=self.hop_time)

            for text_data_path in tqdm(
                glob(join(tok_path, "*.json")),
                desc=f"F0 {builder.NAME}",
            ):
                text_data = read_json(text_data_path)
                name = basename(text_data_path).replace(".json", "")
                audio_path = builder.get_audio_path(name)

                # load audio
                waveform, tmp_sr = torchaudio.load(builder.get_audio_path(name))
                if tmp_sr != self.sr:
                    resampler = AT.Resample(orig_freq=tmp_sr, new_freq=self.sr)
                    waveform = resampler(waveform)

                f0_dict = _F0(waveform)  # f0, mean, std
                f0_dict["sr"] = self.sr
                f0_dict["hop_time"] = self.hop_time

                tmp_f0_path = join(f0_path, name + ".pt")
                torch.save(f0_dict, tmp_f0_path)

                # add/change paths
                text_data["audio"] = audio_path
                write_json(text_data, text_data_path)

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
            # savedir:
            #   builder.root/f0-{self.sr}_hop-{self.hop_time}_win-{self.window_time}
            if self.f0:
                self.preprocess_f0(builder, tok_path)

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
            f0=self.f0,
            rms=self.rms,
            log_rms=self.log_rms,
            normalize_f0=self.normalize_f0,
            interpolate_f0=self.interpolate_f0,
            post_silence=self.post_silence,
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
    def add_data_specific_args(
        parent_parser,
        datasets=["maptask"],
        waveform=False,
        f0=False,
        rms=False,
        log_rms=False,
        normalize_f0=False,
        interpolate_f0=False,
        post_silence=False,
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
        parser.add_argument("--f0", action="store_true", default=f0)
        parser.add_argument("--rms", action="store_true", default=rms)
        parser.add_argument("--log_rms", action="store_true", default=log_rms)
        parser.add_argument("--normalize_f0", action="store_true", default=normalize_f0)
        parser.add_argument(
            "--interpolate_f0", action="store_true", default=interpolate_f0
        )
        parser.add_argument("--post_silence", action="store_true", default=post_silence)

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
    parser = AcousticGPTDM.add_data_specific_args(
        parser, datasets=["maptask"], prosody=True, post_silence=True
    )
    args = parser.parse_args()
    args.hop_time = 0.01
    args.window_time = 0.1
    args.window_time = 0.1
    # args.prosody = True
    # args.num_workers = 0
    dm = AcousticGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    batch = next(iter(dm.val_dataloader()))
    print(len(batch))

    if False:
        dset = AcousticDataset(
            dm.val_filepaths,
            sr=8000,
            word_audio_segment_time=1.0,
            hop_length=400,
            sp1_idx=50257,
            sp2_idx=50258,
            prosody=True,
        )

        input_ids, speaker_ids, audio, prosody = dset[0]

        for d in dset:
            input_ids, speaker_ids, audio, prosody = d
            print("input_ids: ", tuple(input_ids.shape))
            print("prosody: ", tuple(prosody.shape))
            print("audio: ", tuple(audio.shape))

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        for i in range(prosody.shape[0]):
            ax.cla()
            ax.plot(prosody[i, :, 0], label="pitch")
            ax.plot(prosody[i, :, 1], label="voiced")
            ax.plot(prosody[i, :, 2], label="zcr")
            ax.plot(prosody[i, :, 3], label="rms")
            plt.legend()
            plt.pause(0.5)


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
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.batch_size = 16
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.train_dataloader()

    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"]
        speaker_ids = batch["speaker_ids"]
        # print("input_ids: ", tuple(input_ids.shape))
        # print("speaker_ids: ", tuple(speaker_ids.shape))
        # print("-" * 30)

    # loader = dm.val_dataloader()
    # for batch in tqdm(loader, desc="val"):
    #     input_ids = batch["input_ids"]
    #     speaker_ids = batch["speaker_ids"]
    #     print("input_ids: ", tuple(input_ids.shape))
    #     print("speaker_ids: ", tuple(speaker_ids.shape))
    #     print("-" * 30)

    # dm.setup("test")
    # loader = dm.test_dataloader()
    # for batch in tqdm(loader, desc="test"):
    #     input_ids = batch["input_ids"]
    #     speaker_ids = batch["speaker_ids"]
    #     print("input_ids: ", tuple(input_ids.shape))
    #     print("speaker_ids: ", tuple(speaker_ids.shape))
    #     print("-" * 30)

    if False:
        batch = next(iter(loader))
        b = 1
        input_ids = batch["input_ids"][0]
        speaker_ids = batch["speaker_ids"][0]
        print("input_ids: ", tuple(input_ids.shape))
        print("speaker_ids: ", tuple(speaker_ids.shape))
        if "waveform" in batch:
            waveform = batch["waveform"][0]
            print("waveform: ", tuple(waveform.shape))
        if "f0" in batch:
            f0 = batch["f0"][0]
            print("f0: ", tuple(f0.shape))

        fig, ax = plt.subplots(2, 1, figsize=(9, 6))
        for ids, wave, _f0 in zip(input_ids, waveform, f0):
            tok = convert_ids_to_tokens(ids, dm.tokenizer)
            # ff0 = F0_swipe(wave, hop_length=80, sr=8000)
            print(tok[0])
            sd.play(wave, samplerate=dm.sr)
            for a in ax:
                a.cla()
            ax[0].plot(wave)
            ax[0].set_ylim([-0.5, 0.5])
            ax[1].plot(_f0)
            # ax[1].plot(ff0)
            ax[1].set_ylim([-2.0, 2.0])
            # ax[1].set_ylim([0, 250.0])
            plt.pause(0.01)
            input()
            print()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        for batch in tqdm(loader):
            # input_ids, speaker_ids, audio, prosody = batch
            input_ids, speaker_ids, audio, pros, ps = batch
            print("input_ids: ", tuple(input_ids.shape))
            print("speaker_ids: ", tuple(speaker_ids.shape))
            print("audio: ", tuple(audio.shape))
            print("pros: ", tuple(pros.shape))
            input()
            # print("input_ids: ", tuple(input_ids.shape))
            # if isinstance(prosody, torch.Tensor):
            #     print("prosody: ", tuple(prosody.shape))
            # print("audio: ", tuple(audio.shape))

            for p in pros:
                for tp in p:
                    ax.cla()
                    ax.plot(tp[:, 0].unsqueeze(1), label="pitch")
                    # ax.plot(tp[:, 1].unsqueeze(1), label="voiced")
                    # ax.plot(tp[:, 3].unsqueeze(1), label="zcr")
                    ax.plot(tp[:, 3].unsqueeze(1), label="rms")
                    ax.legend()
                    plt.pause(0.001)
                    input()

        p = pros[0]  # B, T, 1
        p = pros[0, :, :, 0]  # B, T
        v = pros[0, :, :, 1]  # B, T

        from ttd.utils import find_island_idx_len

        idx, dur, val = find_island_idx_len(v[0])

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

from librosa.feature import zero_crossing_rate as librosa_zero_crossing_rate
from pysptk import swipe, rapt


def F0_swipe(
    waveform,
    hop_length=None,
    sr=None,
    hop_time=None,
    f_min=60,
    f_max=240,
    threshold=0.3,
):
    if hop_length is not None:
        hopsize = hop_length
    else:
        hopsize = int(sr * hop_time)

    if waveform.ndim == 1:
        return torch.from_numpy(
            swipe(
                waveform.contiguous().double().numpy(),
                fs=sr,
                hopsize=hopsize,
                min=f_min,
                max=f_max,
                threshold=threshold,
                otype="f0",
            )
        ).float()
    elif waveform.ndim == 2:  # (B, N)
        f0 = []
        for audio in waveform:
            f0.append(
                torch.from_numpy(
                    swipe(
                        audio.contiguous().double().numpy(),
                        fs=sr,
                        hopsize=hopsize,
                        min=f_min,
                        max=f_max,
                        threshold=threshold,
                        otype="f0",
                    )
                ).float()
            )
        return torch.stack(f0)


def root_mean_square(y, step, log=True):
    """
    rms (Root Mean Square) of waveform

    :param y:               torch.Tensor, waveform
    :param window_length:   int
    :param hop_length:      int
    :param log:             bool, to return log(rms)

    Based on librosa.feature.rms for waveform calculation
    ```
    if y is not None:
        y = to_mono(y)
        if center:
            y = np.pad(y, int(frame_length // 2), mode=pad_mode)

        x = util.frame(y, frame_length=frame_length, hop_length=hop_length)

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=0, keepdims=True)
    ```

    take the absolute value for each sample, square and take the average, (optional) apply log
    """

    """
    Time dimension is always on 1 except for (T,) shapes
    """
    if y.ndim > 1:
        y_frames = y.unfold(1, size=step, step=step)
        mod = y.shape[1] % step
        if y.ndim == 2:  # (B, n_samples)
            # y_frames (B, T, step)
            if mod != 0:
                last_frame = torch.zeros_like(y_frames[:, 0])  # (B, step)
                # the last remaining mod-samples in y
                # should be included in the last frame step
                # does not matter where
                last_frame[:, :mod] = y[:, -mod:]
                y_frames = torch.cat((y_frames, last_frame.unsqueeze(1)), dim=1)
        elif y.ndim == 3:
            # y_frames (B, T, C, step)
            if mod != 0:
                last_frame = torch.zeros_like(y_frames[:, 0])  # (B, C, step)
                last_frame[:, :, :mod] = y[:, -mod:].permute(0, 2, 1)
                y_frames = torch.cat((y_frames, last_frame.unsqueeze(1)), dim=1)
    else:
        y_frames = y.unfold(0, size=step, step=step)
        mod = y.shape[0] % step
        if mod != 0:
            last_frame = torch.zeros_like(y_frames[0])  # (step)
            last_frame[:mod] = y[-mod:]
            y_frames = torch.cat((y_frames, last_frame.unsqueeze(0)), dim=0)
    power = y_frames.abs().pow(2).mean(dim=-1)
    if log:
        power = torch.log(power + 1e-10)
    return power


def zero_crossing_rate(y, window_length=2048, hop_length=512, center=True):
    """
    wrapper to handle that input to librosa must be Mono and np.ndarray.
    """
    if y.ndim == 1:
        return (
            torch.from_numpy(
                librosa_zero_crossing_rate(
                    y=y.numpy(),
                    frame_length=window_length,
                    hop_length=hop_length,
                    center=center,
                )
            )
            .squeeze(0)
            .float()
        )
    elif y.ndim == 2:  # (B, N) or (2, N)
        zcrs = []
        for audio in y:
            zcrs.append(
                torch.from_numpy(
                    librosa_zero_crossing_rate(
                        y=audio.numpy(),
                        frame_length=window_length,
                        hop_length=hop_length,
                        center=center,
                    )
                ).float()
            )
        return torch.cat(zcrs)
    else:  # (B, 2, N)
        raise NotImplementedError("Not implemented for ndim==3")


def z_normalize_pitch(f0, voiced, eps=1e-6):
    if f0.ndim > 1:  # (B, T)
        norm_f0 = []
        for f, v in zip(f0, voiced):
            if v.sum() < 2:
                tmp_f0 = torch.zeros_like(f)
            else:
                m = f[v == 1].mean()
                s = f[v == 1].std()
                tmp_f0 = f - m
                tmp_f0 = tmp_f0 / (s + eps)
                tmp_f0[v == 0] = 0
            norm_f0.append(tmp_f0)
        norm_f0 = torch.stack(norm_f0)
    else:
        m = f0[voiced == 1].mean()
        s = f0[voiced == 1].std()
        norm_f0 = (f0 - m) / (s + eps)
        norm_f0[voiced == 0] = 0
    return norm_f0


class Prosody(object):
    def __init__(
        self,
        sr=8000,
        hop_time=0.01,
        window_time=0.1,
        f0_min=60,
        f0_max=300,
        f0_threshold=0.3,
        f0_interpolate=True,
        f0_log=True,
        normalize=True,
    ):
        self.sr = sr
        self.hop_time = hop_time
        self.window_time = window_time
        self.hop_length = int(hop_time * sr)
        self.window_length = int(window_time * sr)
        self.normalize = normalize

        # F0 specific
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_threshold = f0_threshold
        self.f0_interpolate = f0_interpolate
        self.f0_log = f0_log

        self.eps = 1e-6

    def fix_size_last_dim(self, x, N):
        if x.shape[-1] != N:
            diff = N - x.shape[-1]
            if diff > 0:
                if x.ndim > 1:
                    x = torch.cat((x, x[:, -1].unsqueeze(-1)), dim=-1)
                else:
                    x = torch.cat((x, x[-1]), dim=-1)
            else:  # diff is negative
                x = x[..., :diff]
        return x

    def __repr__(self):
        attrs = f"(hop_time={self.hop_time}, window_time={self.window_time}"
        attrs += f", hop_length={self.hop_length}, window_length={self.window_length}"
        attrs += f", f0_min={self.f0_min}, f0_max={self.f0_max}"
        attrs += ", sr={})".format(self.sr)
        return self.__class__.__name__ + attrs

    @staticmethod
    def z_normalize(x, eps=1e-6):
        mm = x.mean(dim=1).unsqueeze(1)
        mu = x.std(dim=1).unsqueeze(1)
        x = (x - mm) / (mu + eps)
        return x

    @staticmethod
    def z_normalize_pitch(f0, voiced, eps=1e-6):
        """
        z normalizes over voiced frames.
        uses the minimum value of normalized pitch for unvoiced frames.
        most likely followed by forward interpolation
        """
        if f0.ndim > 1:  # (B, T)
            norm_f0 = []
            for f, v in zip(f0, voiced):
                if v.sum() < 2:
                    tmp_f0 = torch.zeros_like(f)
                else:
                    m = f[v == 1].mean()
                    s = f[v == 1].std()
                    tmp_f0 = f - m
                    tmp_f0 = tmp_f0 / (s + eps)
                    tmp_f0[v == 0] = 0
                    # if torch.isnan(tmp_f0.sum()):
                    #     print(m, s)
                    #     print(tmp_f0)
                    #     input()
                    # print(tmp_f0.max())
                norm_f0.append(tmp_f0)
            norm_f0 = torch.stack(norm_f0)
        else:
            m = f0[voiced == 1].mean()
            s = f0[voiced == 1].std()
            norm_f0 = (f0 - m) / (s + eps)
            norm_f0[voiced == 0] = 0
        return norm_f0

    @staticmethod
    def normalize(p):
        """
        Expexts tensor of shape (2, N, 4): (channels, frames, [pitch, voiced, zcr, rms])
        """
        # pitch
        p[..., 0] = Prosody.z_normalize_pitch(p[..., 0], p[..., 1])
        p[..., 2] = Prosody.z_normalize(p[..., 2])
        p[..., 3] = Prosody.z_normalize(p[..., 3])
        return p

    def interpolate_forward(self, f0, voiced, start_vals=None):
        new_x = torch.zeros_like(f0)
        if f0.ndim == 1:
            if start_vals is not None:
                p_last = start_vals
            else:
                p_last = f0[0]

            for i, (p, v) in enumerate(f0, voiced):
                if v:
                    new_x[i] = p
                    p_last = p
                else:
                    new_x[i] = p_last
        elif f0.ndim == 2:
            for batch in range(f0.shape[0]):
                if start_vals is not None:
                    p_last = start_vals
                else:
                    p_last = f0[batch, 0]
                for i, (p, v) in enumerate(zip(f0[batch], voiced[batch])):
                    if v:
                        new_x[batch, i] = p
                        p_last = p
                    else:
                        new_x[batch, i] = p_last
        else:
            raise NotImplementedError("not implemented for x.ndim > 2")
        return new_x

    def __call__(self, y, vad=None):
        """
        :param y:   torch.Tensor, waveform

        Return:
            proso,  torch.Tensor, (B, 4, N) lf0, voiced, rms (log), zcr
        """
        max_frames = int(y.shape[-1] // self.hop_length) + 1

        # F0
        f0 = F0_swipe(
            y,
            sr=self.sr,
            hop_time=self.hop_time,
            f_min=self.f0_min,
            f_max=self.f0_max,
            threshold=self.f0_threshold,
        )
        voiced = (f0 != 0) * 1

        f0 = self.fix_size_last_dim(f0, max_frames)
        voiced = self.fix_size_last_dim(voiced, max_frames)

        if vad is not None:
            voiced[vad == 0] = 0

        if self.f0_log:
            f0[voiced == 1] = torch.log(f0[voiced == 1] + self.eps)  # log voiced

        # Normalize
        if self.normalize:
            # only consider where defined (voiced frames)
            # make all unvoiced frames the lowest value
            f0 = Prosody.z_normalize_pitch(f0, voiced)

        # print("f0: ", tuple(f0.shape))
        # print("voiced: ", tuple(voiced.shape))
        # # Fill min
        # f0[voiced == 0] == f0[0].min()

        if self.f0_interpolate:
            # interpolate values forward until next new value
            f0 = self.interpolate_forward(f0, voiced)

        # ZeroCrossingRate
        zcr = zero_crossing_rate(
            y, window_length=self.hop_length, hop_length=self.hop_length
        )
        zcr = self.fix_size_last_dim(zcr, max_frames)

        rms = root_mean_square(y, step=self.hop_length, log=False)
        rms = self.fix_size_last_dim(rms, max_frames)
        # if self.normalize:
        #     rms = self.z_normalize(rms)

        proso = torch.stack((f0, voiced, zcr, rms), dim=-1)
        return proso.float()


class AcousticDataset(Dataset):
    def __init__(
        self,
        filepaths,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        sr=8000,
        word_audio_segment_time=1.0,
        hop_length=80,
        n_fft=800,
        prosody=False,
        chunk_size=128,
    ):
        self.filepaths = filepaths
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx

        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.word_audio_segment_time = word_audio_segment_time
        self.n_samples = int(sr * word_audio_segment_time)
        self.n_frames = self.n_samples // self.hop_length
        self.prosody = prosody
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        text_data_path, pros_path = self.filepaths[idx]
        data = read_json(text_data_path)

        speaker_ids = torch.tensor(data["speaker_ids"])
        input_ids = torch.tensor(data["input_ids"])
        channel = (speaker_ids == self.sp2_idx).long()
        offset = int(min(data["starts"]) * self.sr)
        ends = (torch.tensor(data["ends"]) * self.sr).round().long() - offset

        pros = []
        if self.prosody:
            prosody_data = torch.load(pros_path)
            assert self.sr == prosody_data["sr"], "sr must match sr in prosody"
            hop_length = prosody_data["hop_length"]
            assert (
                self.hop_length == hop_length
            ), f"hop_length must match hop_length in prosody, {hop_length}!={self.hop_length}"
            n_fft = prosody_data["n_fft"]
            assert (
                self.n_fft == n_fft
            ), f"n_fft must match n_fft in prosody, {n_fft}!={self.n_fft}"

            # normalize prosody
            prosody = prosody_data["prosody"]  # prosody for entire file
            prosody = Prosody.normalize(prosody)

            frame_offset = librosa.samples_to_frames(
                offset, hop_length=self.hop_length, n_fft=self.n_fft
            )
            frame_ends = librosa.samples_to_frames(
                ends, hop_length=self.hop_length, n_fft=self.n_fft
            )

        waveform, sr = torchaudio.load(data["audio"])
        assert sr == self.sr

        audio = []
        for i, (ch, end_sample) in enumerate(zip(channel, ends)):
            start_sample = max(end_sample - self.n_samples, 0)
            wav = waveform[ch, start_sample:end_sample]
            if len(wav) != self.n_samples:
                wav = torch.cat((torch.zeros(self.n_samples - len(wav)), wav))
            audio.append(wav)

            if self.prosody:
                end_frame = librosa.samples_to_frames(
                    end_sample, hop_length=self.hop_length, n_fft=self.n_fft
                )
                start_frame = librosa.samples_to_frames(
                    start_sample, hop_length=self.hop_length, n_fft=self.n_fft
                )
                p = prosody[ch, start_frame:end_frame]
                if len(p) != self.n_frames:
                    p = torch.cat((torch.zeros(self.n_frames - len(p), p.size(-1)), p))
                pros.append(p)

        audio = torch.stack(audio)
        assert len(input_ids) == len(speaker_ids) == len(audio), "data not same length"

        if self.prosody:
            pros = torch.stack(pros)
            assert len(pros) == len(
                input_ids
            ), f"data not same length {pros.shape}, {input_ids.shape}"

        if self.chunk_size > 0:
            if len(input_ids) != self.chunk_size:
                diff = self.chunk_size - len(input_ids)
                input_ids = torch.cat((input_ids, torch.tensor([self.pad_idx] * diff)))
                speaker_ids = torch.cat(
                    (speaker_ids, torch.tensor([self.pad_idx] * diff))
                )

            if len(audio) != self.chunk_size:
                diff = self.chunk_size - len(audio)
                audio = torch.cat((audio, torch.zeros((diff, *audio.shape[1:]))))

            if self.prosody:
                if len(pros) != self.chunk_size:
                    diff = self.chunk_size - len(pros)
                    pros = torch.cat((pros, torch.zeros((diff, *pros.shape[1:]))))

        # return {"input_ids": input_ids, "speaker_ids": speaker_ids, "audio": audio}
        return input_ids, speaker_ids, audio, pros


class AcousticGPTDM(pl.LightningDataModule):
    def __init__(self, hparams, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)
        self.hparams = hparams

        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_workers"]

        # Tokenizer

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

        # audio
        self.sr = hparams["sample_rate"]
        self.hop_time = hparams["hop_time"]
        self.hop_length = int(self.sr * self.hop_time)
        self.window_time = hparams["window_time"]
        self.n_fft = int(self.sr * self.window_time)
        self.word_audio_segment_time = hparams["word_audio_segment_time"]
        self.prosody = hparams["prosody"]
        self.word_window_size = int(self.sr * self.word_audio_segment_time)

        # builder
        self.builders = create_builders(hparams)

    def get_prosody_path(self, root):
        prosody_path = join(root, "prosody")
        prosody_path += f"_sr-{self.sr}_hop-{self.hop_time}_win-{self.window_time}"
        return prosody_path

    def get_session_name(self, name):
        return name.replace(".json", "").split("_#")[0]

    def prepare_prosody(self, builder, tok_path):
        """
        Extracts Un-normalized prosody for the dialog audio and saves to pros path
        """
        prosody_path = self.get_prosody_path(builder.root)
        print(prosody_path)

        if not exists(prosody_path) or len(listdir(prosody_path)) < 10:
            makedirs(prosody_path, exist_ok=True)
            prosody_encoder = Prosody(
                self.sr,
                hop_time=self.hop_time,
                window_time=self.window_time,
                f0_min=60,
                f0_max=300,
                f0_threshold=0.3,
                f0_interpolate=False,
                f0_log=False,
                normalize=False,
            )

            for text_data_path in tqdm(
                glob(join(tok_path, "*.json")),
                desc=f"Prosody segments {builder.NAME}",
            ):
                text_data = read_json(text_data_path)
                name = basename(text_data_path).replace(".json", "")
                audio_path = builder.get_audio_path(name)

                # load audio
                waveform, tmp_sr = torchaudio.load(builder.get_audio_path(name))
                if tmp_sr != self.sr:
                    resampler = AT.Resample(orig_freq=tmp_sr, new_freq=self.sr)
                    waveform = resampler(waveform)

                prosody = prosody_encoder(waveform)
                pros = {
                    "sr": self.sr,
                    "hop_length": self.hop_length,
                    "n_fft": self.n_fft,
                    "prosody": prosody,
                }
                pros_path = join(prosody_path, name + ".pt")
                torch.save(pros, pros_path)

                text_data["prosody"] = pros_path
                text_data["audio"] = audio_path
                write_json(text_data, text_data_path)

    def prepare_chunked_audio(self, builder, chunked_path):
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
                desc=f"Add audio segments {builder.NAME}",
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

            # prepares prosody for each audio file
            # savedir:
            #   builder.root/prosody_sr-{self.sr}_hop-{self.hop_time}_win-{self.window_time}
            self.prepare_prosody(builder, tok_path)

            # prepare chunks
            chunked_path = builder.prepare_chunked_tokens(
                tok_path,
                chunk_size=self.hparams["chunk_size"],
                overlap=self.hparams["chunk_overlap"],
                keep_length=self.hparams["chunk_keep_length"],
            )

            # splits audio for tokenized chunks for faster load
            self.prepare_chunked_audio(builder, chunked_path)

            # we get more files after chunking then dialogs we defined in the splits
            # this changes the splits to include all chunks
            builder.task_path = chunked_path  # used in dm.setup()
            builder.transform_split_filepaths_with_chunks(builder.task_path)

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            self.train_filepaths = []
            self.val_filepaths = []

            for builder in self.builders:
                pros_root = self.get_prosody_path(
                    builder.root
                )  # for current sr, hop, window size
                for json_name in builder.train_filepaths:
                    # task path points to folder for current task
                    # e.g. root/tokenized_turn_level_explicit_turns_chunk-128
                    text_path = join(builder.task_path, json_name)
                    name = self.get_session_name(json_name)
                    pros_path = join(pros_root, name + ".pt")
                    self.train_filepaths.append((text_path, pros_path))

                for json_name in builder.val_filepaths:
                    text_path = join(builder.task_path, json_name)
                    name = self.get_session_name(json_name)
                    pros_path = join(pros_root, name + ".pt")
                    self.val_filepaths.append((text_path, pros_path))

            self.train_dset = AcousticDataset(
                self.train_filepaths,
                sr=self.sr,
                word_audio_segment_time=self.word_audio_segment_time,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                prosody=self.prosody,
                chunk_size=self.hparams["chunk_size"],
            )
            self.val_dset = AcousticDataset(
                self.val_filepaths,
                sr=self.sr,
                word_audio_segment_time=self.word_audio_segment_time,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                prosody=self.prosody,
                chunk_size=self.hparams["chunk_size"],
            )

        if stage == "test":
            self.test_filepaths = []
            for builder in self.builders:
                for json_name in builder.test_filepaths:
                    name = self.get_session_name(json_name)
                    pros_path = join(pros_root, name + ".pt")
                    self.test_filepaths.append((text_path, pros_path))

            self.test_dset = AcousticDataset(
                self.test_filepaths,
                sr=self.sr,
                word_audio_segment_time=self.word_audio_segment_time,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                prosody=self.prosody,
                chunk_size=self.hparams["chunk_size"],
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
    def add_data_specific_args(parent_parser, datasets=None):
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

        # Audio
        parser.add_argument("--word_audio_segment_time", type=float, default=1)
        parser.add_argument("--sample_rate", type=int, default=8000)
        parser.add_argument("--hop_time", type=float, default=0.05)
        parser.add_argument("--window_time", type=float, default=0.1)
        parser.add_argument("--prosody", action="store_true", default=False)

        if datasets is None:
            parser.add_argument(
                "--datasets",
                nargs="*",
                type=str,
                default=["coached"],
            )
        else:
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
    from ttd.basebuilder import add_builder_specific_args

    parser = ArgumentParser()
    parser = AcousticGPTDM.add_data_specific_args(parser, datasets=["maptask"])
    args = parser.parse_args()
    # args.prosody = True
    # args.num_workers = 0
    dm = AcousticGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    loader = dm.train_dataloader()
    for batch in tqdm(loader):
        input_ids, speaker_ids, audio, prosody = batch
        # print("input_ids: ", tuple(input_ids.shape))
        # if isinstance(prosody, torch.Tensor):
        #     print("prosody: ", tuple(prosody.shape))
        # print("audio: ", tuple(audio.shape))

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

import torch
import librosa
from pysptk import swipe, rapt

from ttd.utils import find_island_idx_len


def F0_swipe(
    waveform,
    hop_length=None,
    sr=None,
    hop_time=None,
    f_min=60,  # default in swipe
    f_max=240,  # default in swipe
    threshold=0.5,  # custom defualt (0.3 in swipe)
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


def fix_size_last_dim(x, N):
    """
    Make sure that the length of the last dim of x is N.
    If N is longer we append the last value if it is shorter we
    remove superfluous frames.
    """
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


def clean_f0(f0, f_min=None):
    """
    removes single valued frames i.e. [0, 0, 123, 0 0] -> [0, 0, 0, 0, 0]. The f0-extractor is sensitive and sometimes
    classifies noise as f0.
    """

    # the minimum value for f0 is sometimes triggered (even for 10-60Hz) and seems to be mostly due to noise
    # they seem very out of distribution compared to all the regular f0 peaks and valleys.
    # the are constant for several frames
    if f_min is not None:
        f0[f0 == f_min] = 0

    if f0.ndim == 1:
        discrete = (f0 > 0).float()
        idx, dur, val = find_island_idx_len(discrete)
        dur = dur[val == 1]  # all duration 1
        idx = idx[val == 1][dur == 1]  # index for duration 1
        f0[idx] = 0
    elif f0.ndim == 2:
        for i, ch_f0 in enumerate(f0):
            discrete = (ch_f0 > 0).float()
            idx, dur, val = find_island_idx_len(discrete)
            dur = dur[val == 1]  # all duration 1
            idx = idx[val == 1][dur == 1]  # index for duration 1
            f0[i, idx] = 0
    return f0


def pitch_statistics(f0):
    voiced = (f0 > 0).float()
    if f0.ndim == 1:
        m = f0[voiced == 1].mean()
        s = f0[voiced == 1].std()
    else:
        m = []
        s = []
        for f, v in zip(f0, voiced):
            m.append(f[v == 1].mean())
            s.append(f[v == 1].std())
        m = torch.stack(m, dim=-1)
        s = torch.stack(s, dim=-1)
    return m, s


def f0_z_normalize(f0, mean, std, eps=1e-8):
    assert mean.size() == std.size()
    voiced = f0 > 0
    n = f0.clone()
    if f0.ndim == 1:
        n[voiced] = (f0[voiced] - mean) / (std + eps)
    else:
        for i, (f, v) in enumerate(zip(f0, voiced)):
            n[i, v] = (f[v] - mean[i]) / (std[i] + eps)
    return n


def interpolate_forward_slow(f0, voiced, start_vals=None):
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


def interpolate_forward(f0, voiced):
    f = f0.clone()
    if f0.ndim == 1:
        v = voiced
        idx, dur, val = find_island_idx_len(v.float())
        # unvoiced -> value prior unvoiced
        dur = dur[val == 0]
        idx = idx[val == 0]
        for ii, dd in zip(idx, dur):
            if ii - 1 < 0:
                tmp_val = f[0]
            else:
                tmp_val = f[ii - 1]
            f[ii : ii + dd] = tmp_val
    else:
        for i, v in enumerate(voiced):
            idx, dur, val = find_island_idx_len(v.float())
            # unvoiced -> value prior unvoiced
            dur = dur[val == 0]
            idx = idx[val == 0]
            for ii, dd in zip(idx, dur):
                if ii - 1 < 0:
                    tmp_val = f[i, 0]
                else:
                    tmp_val = f[i, ii - 1]
                f[i, ii : ii + dd] = tmp_val
    return f


class F0(object):
    def __init__(
        self,
        sr=8000,
        hop_time=0.01,
        f0_min=60,
        f0_max=300,
        f0_threshold=0.5,
    ):
        self.sr = sr
        self.hop_time = hop_time
        self.hop_length = int(hop_time * sr)

        # F0 specific
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_threshold = f0_threshold
        self.eps = 1e-8

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tsr={self.sr},"
        s += f"\n\thop_time={self.hop_time},"
        s += f"\n\tf0_min={self.f0_min},"
        s += f"\n\tf0_max={self.f0_max},"
        s += f"\n\tf0_threshold={self.f0_threshold},"
        s += f"\n\tf0_max={self.f0_max},"
        return s

    def __call__(self, waveform):
        """
        :param y:   torch.Tensor, waveform (n_samples,) or (B, n_samples)
        Return:
            dict,  f0, mean, std
        """
        n_frames = int(waveform.shape[-1] // self.hop_length) + 1

        f0 = F0_swipe(
            waveform,
            sr=self.sr,
            hop_time=self.hop_time,
            threshold=self.f0_threshold,
            f_min=self.f0_min,
        )
        f0 = fix_size_last_dim(f0, n_frames)
        f0 = clean_f0(f0, f_min=self.f0_min)
        m, s = pitch_statistics(f0)
        return {"f0": f0, "mean": m, "std": s}


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torchaudio
    import torchaudio.transforms as AT
    import time

    sr = 8000
    hop_time = 0.01
    window_time = 0.1
    hop_length = int(sr * hop_time)
    window_length = int(sr * window_time)

    # Load waveform + resample
    wavpath = "data/maptask/audio/q2ec6.wav"
    waveform, tmp_sr = torchaudio.load(wavpath)
    if tmp_sr != sr:
        resampler = AT.Resample(orig_freq=tmp_sr, new_freq=sr)
        waveform = resampler(waveform)
    print("waveform: ", tuple(waveform.shape))  # CH, n_samples

    _F0 = F0()

    f0_dict = _F0(waveform)

    f0 = f0_dict["f0"]
    f0n = f0_z_normalize(f0, mean=f0_dict["mean"], std=f0_dict["std"])

    voiced = f0 > 0

    n = 10
    t = time.time()
    for i in range(n):
        f0i = interpolate_forward_slow(f0, voiced)
        f0ni = interpolate_forward_slow(f0n, voiced)
    print(round((time.time() - t) / n, 3))
    t = time.time()
    for i in range(n):
        f0i2 = interpolate_forward(f0, voiced)
        f0ni2 = interpolate_forward(f0n, voiced)
    print(round((time.time() - t) / n, 3))

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    N = int(sr / hop_length)
    for i in range(0, f0.shape[-1], N):
        for a in ax:
            a.cla()
        ax[0].set_ylim([0, 300])
        ax[1].set_ylim([-1, 1])
        ax[0].plot(f0[0, i : i + N * 2], alpha=0.5, linewidth=1)
        ax[0].plot(f0i[0, i : i + N * 2])
        ax[0].plot(f[0, i : i + N * 2])
        ax[1].plot(f0n[0, i : i + N * 2], alpha=0.5, linewidth=1)
        ax[1].plot(f0ni[0, i : i + N * 2])
        # ax[1].plot(f0_int_z[0, i : i + N * 2])
        plt.pause(0.01)
        input()

    # remove single values

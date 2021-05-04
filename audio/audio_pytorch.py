import math
import torch
import torch.nn as nn
import torchaudio.transforms as AT
import torchaudio.functional as AF

from turngpt.F0 import F0, f0_z_normalize, interpolate_forward, F0_swipe


class Pitch(nn.Module):
    def __init__(self, sr=8000, hop_time=0.01, f0_min=60, f0_max=400, f0_threshold=0.2):
        super().__init__()
        self.sr = sr
        self.hop_time = hop_time
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_threshold = f0_threshold

        self._F0 = F0(
            sr=sr,
            hop_time=hop_time,
            f0_min=f0_min,
            f0_max=f0_max,
            f0_threshold=f0_threshold,
        )

    def forward(self, x):
        reshape = False
        if x.ndim == 3:
            B, N, n_samples = x.size()
            reshape = True
            x = x.view(-1, n_samples)

        f0 = []
        for tmp_x in x:
            f0.append(self._F0(tmp_x.unsqueeze(0))["f0"][0, :-1])
        f0 = torch.stack(f0)

        if reshape:
            f0 = f0.contiguous().view(B, N, -1)
        return f0


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


if __name__ == "__main__":

    from argparse import ArgumentParser
    from turngpt.acousticDM import AudioDM
    from ttd.tokenizer_helpers import convert_ids_to_tokens
    import matplotlib.pyplot as plt
    import sounddevice as sd

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
        log_rms=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.batch_size = 4
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()

    batch = next(iter(loader))

    w = batch["waveform"]
    tokens = convert_ids_to_tokens(batch["input_ids"], dm.tokenizer)
    f = batch["f0"]

    pitcher = Pitch()
    p = pitcher(w)

    pp = []
    for _p in p:
        pp.append(interpolate_forward(_p, _p != 0))
    pp = torch.stack(pp)

    gaussian_filter = Gaussian1D(kernel_size=3)
    pg = gaussian_filter(pp.view(-1, pp.shape[-1]).unsqueeze(-2))
    pg = pg.view(*pp.shape)

    from tqdm import tqdm

    _F0 = F0(sr=8000, hop_time=0.01, f0_threshold=0.2)

    new_f = []
    for b in range(f.shape[0]):
        batch_f = []
        for i in tqdm(range(f.shape[1])):
            _f = _F0(w[b, i].unsqueeze(0))["f0"][0, :-1]
            v = _f != 0
            _f = interpolate_forward(_f.unsqueeze(0), v.unsqueeze(0))[0]
            batch_f.append(_f)
        new_f.append(torch.stack(batch_f))
    new_f = torch.stack(new_f)

    b = 0
    fig, ax = plt.subplots(3, 1)
    for i in range(w.shape[1]):
        for a in ax:
            a.cla()
        if i != 127:
            ax[0].set_title(tokens[b, i] + " -> " + tokens[b, i + 1])
        ax[0].plot(w[b, i])
        ax[1].plot(f[b, i])
        ppp = pg[b, i].clone()
        # ppp[ppp!=0] = ppp[ppp!=0].log()
        # ax[2].plot((pp[b, i]+1e-8).log())
        ax[2].plot(ppp)
        # ax[2].set_ylim([2,6])
        ax[2].set_ylim([80, 250])
        # ax[2].plot(new_f[b, i])
        ax[0].set_xlim([0, 8000])
        ax[1].set_xlim([0, 100])
        ax[2].set_xlim([0, 100])
        plt.tight_layout()
        plt.pause(0.01)
        input()

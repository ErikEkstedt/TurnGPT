import torch
import torch.nn as nn

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    TransposeLast,
)
from typing import List, Tuple


MODEL_PATH = "checkpoints/wav2vec2/wav2vec_small.pt"


def total_parameters(model, only_learnable=False, verbose=True):
    def nice_int(n):
        s = ""
        for j, x in enumerate(reversed(str(n))):
            if j % 3 == 0 and j != 0:
                s += ","
            s += x
        return "".join(reversed(s))

    if only_learnable:
        total_param_size = sum(
            p.data.numel() for p in model.parameters() if p.requires_grad
        )
    else:
        total_param_size = sum(p.data.numel() for p in model.parameters())

    if verbose:
        print("params: ", nice_int(total_param_size))
    return total_param_size


def hacky_load(model_path):
    cfg = Wav2Vec2Config()
    cp = torch.load(model_path)

    for k, v in vars(cp["args"]).items():
        if k in vars(cfg):
            setattr(cfg, k, v)
    model = Wav2Vec2Model(cfg)

    model.load_state_dict(cp["model"])
    return model, cfg


class Wav2Vec2Encoder(nn.Module):
    """
    Used as normal:
        enc = Wav2VecEncoder(·)
        normal_output = enc.model(source, padding_mask, mask, features_only)


    Used as Encoder:

    ```python
        enc = Wav2VecEncoder(·)
        enc.freeze()  # freeze weights
        out = enc(x)
        out['x']
        out['padding_mask']
    ```

    """

    def __init__(self, cfg=None, model_path=None):
        super().__init__()
        if model_path is not None:
            self.model, self.cfg = hacky_load(model_path)
        else:
            assert cfg is not None, "must provide cfg if no 'model_path'"
            self.cfg = cfg
            self.model = Wav2Vec2Model(cfg)

        self.output_size = self.cfg.encoder_embed_dim

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        print("Froze Wav2Vec weights...")

    @property
    def nparams(self):
        return total_parameters(self, only_learnable=False, verbose=False)

    @property
    def device(self):
        return self.model.feature_extractor.conv_layers[0][0].weight.device

    def forward(self, source, padding_mask=None, mask=True, features_only=False):
        with torch.no_grad():
            out = self.model(x, padding_mask=None, mask=False, features_only=True)
        return out


class W2VCNNProjection(nn.Module):
    """
    a (very) Slightly  modified version of fairseq-wav2vec2 "ConvFeatureExtractionModel" cnn encoder

    SOURCE:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec/wav2vec2.py


    CNN projection E: z = wav2vec2(x)  -> z_small

    :params:
        conv_layers: List[Tuple[int, int, int]],  [(hidden_dim, kernel, stride), ... ]
        in_channels: int=768,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    """

    def __init__(
        self,
        in_channels: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}
        in_d = in_channels
        # self.conv_layers = nn.ModuleList()
        self.conv_layers = []
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            self.conv_layers.append(
                self._block(
                    n_in=in_d,
                    n_out=dim,
                    k=k,
                    stride=stride,
                    dropout=dropout,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)

    def _block(
        self,
        n_in,
        n_out,
        k,
        stride,
        dropout,
        is_layer_norm=False,
        is_group_norm=False,
        conv_bias=False,
    ):
        def make_conv():
            conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            return conv

        assert (
            is_layer_norm and is_group_norm
        ) == False, "layer norm and group norm are exclusive"

        if is_layer_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                nn.Sequential(
                    TransposeLast(),
                    Fp32LayerNorm(n_out, elementwise_affine=True),
                    TransposeLast(),
                ),
                nn.GELU(),
            )
        elif is_group_norm:
            return nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout),
                Fp32GroupNorm(n_out, n_out, affine=True),
                nn.GELU(),
            )
        else:
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        x = x.permute(0, 2, 1).contiguous()  # BTD -> BDT
        for conv in self.conv_layers:
            x = conv(x)
        x = x.permute(0, 2, 1).contiguous()  # BDT -> BTD
        return x


if __name__ == "__main__":

    from argparse import ArgumentParser
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=False,
        waveform=True,
        rms=False,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.sample_rate = 16000
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    model = Wav2Vec2Encoder(model_path=MODEL_PATH).to("cuda")
    model.freeze()
    model.nparams

    proj_in = model.output_size

    kernel_size = [3, 3, 3, 3]
    stride = [2, 2, 2, 2]
    hidden = [256, 256, 256, 128]
    conv_layers = [(hidden[0], kernel_size[0], stride[0])]  # [(dim, kernel, stride)]
    for i in range(1, len(kernel_size)):
        conv_layers.append((hidden[i], kernel_size[i], stride[i]))
    proj = W2VCNNProjection(in_channels=proj_in, conv_layers=conv_layers)

    z = proj(out["x"])
    print(z.shape)

    batch = next(iter(dm.train_dataloader()))
    batch_size = 32
    x = batch["waveform"]  # B, N, 16000
    # x = x.view(-1, x.shape[-1]) # B*N, 16000
    # x = x[0]
    x = x[0, :batch_size]  # N, 16000

    wav_encoding = model.encode(x.to(model.device))

    m = nn.MaxPool1d(5, stride=1, padding=0)
    input = torch.randn(20, 16, 5)
    output = m(input)
    print(output.shape)

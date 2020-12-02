import torch
import torch.nn as nn


def conv_output_size(input_size, kernel_size, padding, stride):
    return int(((input_size - kernel_size + 2 * padding) / stride) + 1)


class SPFConv(nn.Module):
    def __init__(
        self,
        frames=20,
        n_feats=4,
        hidden=32,
        output_size=128,
        num_layers=3,
        kernel_size=3,
        first_stride=2,
        independent=True,
        activation="ReLU",
    ):
        super().__init__()
        self.n_feats = n_feats
        self.output_size = output_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.independent = independent
        self.input_frames = frames
        self.out_frames = frames - ((kernel_size - 1) * num_layers)
        self.activation = getattr(nn, activation)
        self.first_stride = first_stride
        self.padding = 0

        if independent:
            self.groups = n_feats
            self.hidden = n_feats * hidden
        else:
            assert hidden is not None, "Must provide `hidden` for joint processing"
            self.groups = 1
            self.hidden = hidden

        self.convs = self._build_layers()
        self.post_conv_frames = self.calc_conv_out()
        assert (
            self.post_conv_frames > 0
        ), f"Dimension too small. Conv out: {self.post_conv_frames}, please change encoder size"
        self.head = nn.Linear(
            self.post_conv_frames * (self.hidden // self.n_feats), output_size
        )
        self.ln = nn.LayerNorm(output_size)

    @property
    def device(self):
        return self.convs[0].weight.device

    def calc_conv_out(self):
        n = conv_output_size(
            self.input_frames, self.kernel_size, self.padding, self.first_stride
        )
        for i in range(self.num_layers - 1):
            n = conv_output_size(n, self.kernel_size, self.padding, 1)
        return n

    def _build_layers(self):
        layers = [
            nn.Conv1d(
                in_channels=self.n_feats,
                out_channels=self.hidden,
                kernel_size=self.kernel_size,
                stride=self.first_stride,
                padding=self.padding,
                groups=self.groups,
            ),
            self.activation(),
        ]
        for i in range(1, self.num_layers):
            layers += [
                nn.Conv1d(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    groups=self.groups,
                ),
                self.activation(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        B, N, T, n_feats = x.size()

        # M: B*N
        z = x.view(-1, T, n_feats)  # N, B, T, n_feats -> M, T, n_feats
        z = z.permute(0, 2, 1)  # M, T, n_feats -> M, n_feats, T
        z = self.convs(z)  # M, n_feats, T -> M, hidden, T

        # separate channel encodings
        z = torch.stack(
            z.chunk(n_feats, dim=1), dim=1
        )  # M, hidden, T -> M, n_feats, hh, T
        z = z.flatten(-2)  # M, n_feats, hh, T -> M, n_feats, hh*T
        z = self.ln(self.head(z))  # M, n_feats, hh*T -> M, n_feats, hidden
        z = z.view(
            B, N, n_feats, self.output_size
        )  # M, n_feats, hidden -> B, N, n_feats, hidden
        return z  # ready to be attended by transformer


if __name__ == "__main__":

    from turngpt.models import gradient_check_batch

    n_feats = 2
    enc = SPFConv(
        frames=100,
        n_feats=n_feats,
        hidden=32,
        output_size=128,
        num_layers=10,
        kernel_size=5,
        first_stride=2,
        independent=True,
        activation="ReLU",
    )
    print(enc)
    print("post conv frames: ", enc.post_conv_frames)
    x = torch.randn((4, 128, 100, n_feats))
    z = enc(x)
    print(z.shape)

    gradient_check_batch(x, enc)

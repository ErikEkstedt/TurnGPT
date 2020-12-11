from argparse import ArgumentParser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from os.path import join

import pytorch_lightning as pl
import matplotlib.pyplot as plt


from turngpt.models import Attention1D
from turngpt.transforms import ClassificationLabelTransform

from matplotlib import use as mpl_use

mpl_use("Agg")
pl.seed_everything(1234)


class SelfAttention(nn.Module):
    """
    Source:
        https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(self, hidden, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert hidden % n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(hidden, hidden)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_frames=100,
        hidden=32,
        in_channels=1,
        num_layers=7,
        kernel_size=3,
        first_stride=2,
        activation="ReLU",
        independent=False,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.in_channels = in_channels
        self.hidden = hidden
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.first_stride = first_stride
        self.padding = 0
        self.activation = getattr(nn, activation)

        if isinstance(self.first_stride, int):
            self.first_stride = [1] * self.num_layers
            self.first_stride[0] = first_stride
        else:
            assert (
                len(self.first_stride) == num_layers
            ), "provide as many strides as layers"

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * self.num_layers
        else:
            assert (
                len(kernel_size) == num_layers
            ), "provide as many kernel_size as layers"

        if independent:
            self.groups = in_channels
            self.hidden = hidden
        else:
            self.groups = 1
            self.hidden = hidden

        self.convs = self._build_conv_layers()
        self.output_size = self.calc_conv_out()

    def conv_output_size(self, input_size, kernel_size, padding, stride):
        return int(((input_size - kernel_size + 2 * padding) / stride) + 1)

    def calc_conv_out(self):
        n = self.input_frames
        for i in range(0, self.num_layers):
            n = self.conv_output_size(
                n, self.kernel_size[i], self.padding, stride=self.first_stride[i]
            )
        return n

    def _build_conv_layers(self):
        layers = [
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.hidden,
                kernel_size=self.kernel_size[0],
                stride=self.first_stride[0],
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
                    kernel_size=self.kernel_size[i],
                    stride=self.first_stride[i],
                    padding=self.padding,
                    groups=self.groups,
                ),
                self.activation(),
            ]
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-2)  # B, T -> B, 1, T
        x = self.convs(x)
        return x


class VectorQuantizerEMA(nn.Module):
    """
    Slightly changed version of Zalando research vq-vae implementation
    Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def perplexity(self, encodings):
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC

        # Flatten input
        B, N, D = inputs.shape
        input_shape = inputs.shape
        flat_input = inputs.contiguous().view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # encodings
        encodings = encodings.view(B, N, self._num_embeddings)
        return quantized, encodings, loss


################################################################################


class ProsodyClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=1,
        conv_hidden=64,
        layers=5,
        kernel=3,
        first_stride=2,
        fc_hidden=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50258,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            num_layers=layers,
            kernel_size=kernel,
            first_stride=first_stride,
            activation=activation,
        )

        fc_in = int(self.conv_encoder.output_size * conv_hidden)

        self.fc = nn.Linear(fc_in, fc_hidden)
        self.head = nn.Linear(fc_hidden, 1)

        self.learning_rate = learning_rate

        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, x):
        x = self.conv_encoder(x)
        x = torch.sigmoid(self.fc(x.flatten(1)))
        x = self.head(x)
        return x

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x = []
        if "f0" in batch:
            x.append(batch["f0"])

        if "rms" in batch:
            x.append(batch["rms"])
        x = torch.stack(x, dim=-1)
        x, y, inp = self.input_to_label(x, batch["input_ids"])
        if x.ndim == 2:
            x = x.squeeze(-2)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)
        return x, y, inp

    def training_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        self.log("val_loss", loss)
        return {"val_loss": loss}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_in_channels", type=int, default=1)
        parser.add_argument("--prosody_conv_hidden", type=int, default=32)
        parser.add_argument("--prosody_conv_layers", type=int, default=5)
        parser.add_argument("--prosody_first_stride", type=int, default=2)
        parser.add_argument("--prosody_kernel", type=int, default=5)
        parser.add_argument("--prosody_fc_hidden", type=int, default=32)
        parser.add_argument("--prosody_activation", type=str, default="ReLU")

        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def experiment(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """
    from turngpt.acousticDM import AudioDM

    parser = ProsodyClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=False,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()

    args.early_stopping = True
    args.patience = 5

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    for chidden in [32, 64, 128]:
        args.prosody_conv_hidden = chidden
        for layers in [5, 7, 9]:
            args.prosody_conv_layers = layers

            name = "ProsConv"
            in_channels = 0
            if args.rms:
                in_channels += 1
                name += "_rms"
            if args.f0:
                in_channels += 1
                name += "_f0"

            print(name)
            print("conv hidden: ", chidden)
            print("conv layers: ", layers)

            model = ProsodyClassifier(
                in_frames=args.prosody_frames,
                in_channels=in_channels,
                conv_hidden=args.prosody_conv_hidden,
                layers=args.prosody_conv_layers,
                kernel=args.prosody_kernel,
                first_stride=args.prosody_first_stride,
                fc_hidden=args.prosody_fc_hidden,
                learning_rate=args.learning_rate,
                activation=args.prosody_activation,
                sp1_idx=dm.sp1_idx,
                sp2_idx=dm.sp2_idx,
                pad_idx=dm.pad_idx,
            )

            logger, checkpoint_callback, callbacks = logging(args, name=name)

            trainer = pl.Trainer.from_argparse_args(
                args,
                logger=logger,
                checkpoint_callback=checkpoint_callback,
                callbacks=callbacks,
            )
            trainer.fit(model, datamodule=dm)


################################################################################
class QClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=1,
        conv_hidden=64,
        layers=5,
        kernel=3,
        first_stride=2,
        fc_hidden=32,
        n_codes=64,
        n_code_dim=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50258,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            num_layers=layers,
            kernel_size=kernel,
            first_stride=first_stride,
            activation=activation,
            independent=False,
        )

        # How to make the codes sparse?
        self.pre_vq = nn.Linear(conv_hidden, n_code_dim)
        self.vq = VectorQuantizerEMA(num_embeddings=n_codes, embedding_dim=n_code_dim)

        n_total_codes = int(n_code_dim * self.conv_encoder.output_size)
        self.fc = nn.Linear(n_total_codes, fc_hidden)
        self.head = nn.Linear(fc_hidden, 1)

        self.learning_rate = learning_rate
        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, x):
        B, H, T = x.size()
        x = self.conv_encoder(x)
        x = x.permute(0, 2, 1)  # (B, H, T) -> (B, T, H)
        x = self.pre_vq(x)  # (B, T, H) -> (B, T, H_vq)
        z_q, enc, vq_loss = self.vq(x)
        # z_q: (B, T, code_dim)
        x = torch.sigmoid(self.fc(x.flatten(1)))
        x = self.head(x)
        return x, vq_loss

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y.float())

    def plot_codes(self, projection, plot=False):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            alpha=0.3,
            label="UMAP discrete code projection",
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    def code_projection(self):
        return umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine").fit_transform(
            self.vq._embedding.weight.data.cpu()
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x = []
        if "f0" in batch:
            x.append(batch["f0"])

        if "rms" in batch:
            x.append(batch["rms"])
        x = torch.stack(x, dim=-1)
        x, y, inp = self.input_to_label(x, batch["input_ids"])
        if x.ndim == 2:
            x = x.squeeze(-2)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)
        return x, y, inp

    def training_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)

        y_pred, vq_loss = self(x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss

        self.log("loss", loss)
        self.log("vq_loss", vq_loss)
        return {"loss": loss}
        return {"loss": loss, "vq_loss": vq_loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred, vq_loss = self(x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss

        self.log("val_loss", loss)
        self.log("val_vq_loss", vq_loss)
        return {"val_loss": loss, "val_vq_loss": vq_loss}

    def test_step(self, batch, batch_idx):
        x, y, input_ids = self.shared_step(batch)
        y_pred, vq_loss = self(x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss
        if batch_idx == 0:
            projection = self.code_projection()
            fig, ax = self.plot_codes(projection)
            fig_path = join(self.logger.log_dir, "projection")
            fig.savefig(fig_path)
            print("saved projection fig -> ", fig_path)
            plt.close()
        return {
            "test_loss": loss,
            "test_vq_loss": vq_loss,
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_in_channels", type=int, default=1)
        parser.add_argument("--prosody_conv_hidden", type=int, default=32)
        parser.add_argument("--prosody_conv_layers", type=int, default=5)
        parser.add_argument("--prosody_first_stride", type=int, default=2)
        parser.add_argument("--prosody_kernel", type=int, default=5)
        parser.add_argument("--prosody_fc_hidden", type=int, default=32)
        parser.add_argument("--prosody_activation", type=str, default="ReLU")

        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def experiment_Q(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """
    from turngpt.acousticDM import AudioDM

    parser = QClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=False,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    args.log_rms = True
    args.rms = True
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    in_channels = 0
    if args.rms:
        in_channels += 1
    if args.f0:
        in_channels += 1

    args.early_stopping = True
    args.patience = 5
    args.prosody_conv_layers = 7
    args.prosody_conv_hidden = 32
    args.prosody_n_code_dim = 32

    for n_codes in [32, 64, 128]:
        name = "ProsQConv"
        in_channels = 0
        if args.rms:
            in_channels += 1
            name += "_rms"
        if args.f0:
            in_channels += 1
            name += "_f0"

        print(name)
        print("n_codes: ", n_codes)
        print("code_dim: ", args.prosody_n_code_dim)
        args.prosody_n_codes = n_codes

        model = QClassifier(
            in_frames=args.prosody_frames,
            in_channels=in_channels,
            conv_hidden=args.prosody_conv_hidden,
            layers=args.prosody_conv_layers,
            kernel=args.prosody_kernel,
            first_stride=args.prosody_first_stride,
            fc_hidden=args.prosody_fc_hidden,
            n_codes=args.prosody_n_codes,
            n_code_dim=args.prosody_n_code_dim,
            learning_rate=args.learning_rate,
            activation=args.prosody_activation,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            pad_idx=dm.pad_idx,
        )
        print(model)

        logger, checkpoint_callback, callbacks = logging(args, name=name)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(test_dataloaders=dm.val_dataloader(), verbose=True)


################################################################################
class QRNNClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=1,
        conv_hidden=64,
        layers=5,
        kernel=3,
        first_stride=2,
        rnn_hidden=96,
        rnn_layers=1,
        n_codes=64,
        n_code_dim=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50258,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            num_layers=layers,
            kernel_size=kernel,
            first_stride=first_stride,
            activation=activation,
            independent=False,
        )

        # How to make the codes sparse?
        self.pre_vq = nn.Linear(conv_hidden, n_code_dim)
        self.vq = VectorQuantizerEMA(num_embeddings=n_codes, embedding_dim=n_code_dim)

        self.rnn = nn.LSTM(
            n_code_dim, rnn_hidden, num_layers=rnn_layers, batch_first=True
        )
        self.head = nn.Linear(rnn_hidden, 1)

        self.learning_rate = learning_rate
        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, x):
        B, H, T = x.size()
        x = self.conv_encoder(x)
        x = x.permute(0, 2, 1)  # (B, H, T) -> (B, T, H)
        x = self.pre_vq(x)  # (B, T, H) -> (B, T, H_vq)
        z_q, enc, vq_loss = self.vq(x)
        # z_q: (B, T, code_dim)
        x, h = self.rnn(z_q)
        x = self.head(x[:, -1])  # last hidden state
        return x, vq_loss

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y.float())

    def plot_codes(self, projection, plot=False):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            alpha=0.3,
            label="UMAP discrete code projection",
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    def code_projection(self):
        return umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine").fit_transform(
            self.vq._embedding.weight.data.cpu()
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x = []
        if "f0" in batch:
            x.append(batch["f0"])

        if "rms" in batch:
            x.append(batch["rms"])
        x = torch.stack(x, dim=-1)
        x, y, inp = self.input_to_label(x, batch["input_ids"])
        if x.ndim == 2:
            x = x.squeeze(-2)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)
        return x, y, inp

    def training_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)

        y_pred, vq_loss = self(x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss

        self.log("loss", loss)
        self.log("vq_loss", vq_loss)
        return {"loss": loss}
        return {"loss": loss, "vq_loss": vq_loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred, vq_loss = self(x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss

        self.log("val_loss", loss)
        self.log("val_vq_loss", vq_loss)
        return {"val_loss": loss, "val_vq_loss": vq_loss}

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        if batch_idx == 0:
            projection = self.code_projection()
            fig, ax = self.plot_codes(projection)
            fig_path = join(self.logger.log_dir, "projection")
            fig.savefig(fig_path)
            print("saved projection fig -> ", fig_path)
            plt.close()
        return {
            "test_loss": metrics["val_loss"],
            "test_vq_loss": metrics["val_vq_loss"],
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_in_channels", type=int, default=1)
        parser.add_argument("--prosody_conv_hidden", type=int, default=32)
        parser.add_argument("--prosody_conv_layers", type=int, default=5)
        parser.add_argument("--prosody_first_stride", type=int, default=2)
        parser.add_argument("--prosody_kernel", type=int, default=5)
        parser.add_argument("--prosody_fc_hidden", type=int, default=32)
        parser.add_argument("--prosody_activation", type=str, default="ReLU")

        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def experiment_QRNN(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """
    from turngpt.acousticDM import AudioDM

    parser = QRNNClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=False,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=True,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    args.log_rms = True
    args.rms = True
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    in_channels = 0
    if args.rms:
        in_channels += 1
    if args.f0:
        in_channels += 1

    args.early_stopping = True
    args.patience = 5
    args.prosody_conv_layers = 7
    args.prosody_conv_hidden = 32
    args.prosody_n_code_dim = 32
    args.prosody_n_codes = 64
    args.prosody_rnn_hidden = 96

    for rnn_layers in [1, 2]:
        args.prosody_rnn_layers = rnn_layers
        name = "ProsQRNNConv"
        in_channels = 0
        if args.rms:
            in_channels += 1
            name += "_rms"
        if args.f0:
            in_channels += 1
            name += "_f0"

        print(name)
        print("rnn_layers: ", rnn_layers)
        print("rnn_hidden: ", args.prosody_rnn_hidden)
        print("code_dim: ", args.prosody_n_code_dim)

        model = QRNNClassifier(
            in_frames=args.prosody_frames,
            in_channels=in_channels,
            conv_hidden=args.prosody_conv_hidden,
            layers=args.prosody_conv_layers,
            kernel=args.prosody_kernel,
            first_stride=args.prosody_first_stride,
            n_codes=args.prosody_n_codes,
            n_code_dim=args.prosody_n_code_dim,
            rnn_hidden=args.prosody_rnn_hidden,
            rnn_layers=args.prosody_rnn_layers,
            learning_rate=args.learning_rate,
            activation=args.prosody_activation,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            pad_idx=dm.pad_idx,
        )
        print(model)

        logger, checkpoint_callback, callbacks = logging(args, name=name)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(test_dataloaders=dm.val_dataloader(), verbose=True)


################################################################################
def logging(args, name="model"):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os import environ
    from os.path import join
    from ttd.utils import get_run_dir

    # Where to save the training
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print("root dir: ", args.save_dir)
    checkpoint_callback = None
    callbacks = None
    logger = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        desc = f"{name} training"
        logger = TensorBoardLogger(
            args.save_dir,
            name=name,
            log_graph=False,
        )
        ch_path = join(logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}-{val_loss:.5f}",
            save_top_k=2,
            mode="min",
            monitor="val_loss",
        )

        # Save the used tokenizer
        # tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        # torch.save(dm.tokenizer, tokenizer_path)
        # print("tokenizer saved -> ", tokenizer_path)

        if args.early_stopping:
            print(f"Early stopping (patience={args.patience})")
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=True,
            )
            callbacks = [early_stop_callback]
        print("-" * 50)
    return logger, checkpoint_callback, callbacks


if __name__ == "__main__":

    # enc = ProsodyClassifier()
    # x = torch.rand((16, 100))
    # y_pred = enc(x)

    parser = ArgumentParser()
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--regular", action="store_true")
    parser.add_argument("--VQ", action="store_true")
    parser.add_argument("--VQRNN", action="store_true")

    tmp_args, _ = parser.parse_known_args()
    if tmp_args.regular:
        experiment(parser)

    if tmp_args.VQ:
        experiment_Q(parser)

    if tmp_args.VQRNN:
        experiment_QRNN(parser)

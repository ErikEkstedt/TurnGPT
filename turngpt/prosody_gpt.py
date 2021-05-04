from argparse import ArgumentParser
import umap
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from turngpt.models.gpt_mini import Block, GPTConfig
from turngpt.models import ConvEncoder, VectorQuantizerEMA
from turngpt.transforms import ClassificationLabelTransform
from turngpt.train_utils import logging
from ttd.tokenizer_helpers import convert_ids_to_tokens


EMB_PATH = "checkpoints/pretrained/PDECTWW/word_embedding_state_dict.pt"


def get_n_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    return total_params


def create_position_codes(n_pos, dim, out):
    """
    SOURCE:
        PARLAI: parlai/agent/transformer/modules.py
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)


class ProsodyGPT(pl.LightningModule):
    def __init__(
        self,
        n_codes=128,
        n_code_dim=300,
        commitment_cost=0.35,
        n_layer=1,
        n_head=1,
        n_tokens=50259,
        n_token_dim=768,
        prosody_frames=100,
        prosody_in_channels=2,
        prosody_conv_hidden=32,
        prosody_kernel=[5, 3, 3, 3, 3, 3],
        prosody_stride=[2, 2, 1, 1, 1, 1],
        prosody_padding=0,
        prosody_activation="ReLU",
        n_pos=128,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sp1_idx = sp1_idx
        self.sp2_idx = sp1_idx
        self.pad_idx = pad_idx

        # Word Embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=n_token_dim
        )

        self.word_projection = nn.Linear(self.word_embedding.embedding_dim, n_code_dim)

        # Acoustic Encoding
        self.acoustic_encoder = ConvEncoder(
            input_frames=prosody_frames,
            hidden=prosody_conv_hidden,
            in_channels=prosody_in_channels,
            num_layers=len(prosody_stride),
            kernel_size=prosody_kernel,
            padding=prosody_padding,
            strides=prosody_stride,
            activation=prosody_activation,
        )
        self.acoustic_projection = nn.Linear(
            self.acoustic_encoder.output_size * prosody_conv_hidden, n_code_dim
        )

        # Combine embedding
        self.modal_act = torch.sigmoid
        self.pre_vq = nn.Linear(2 * n_code_dim, n_code_dim)
        self.vq = VectorQuantizerEMA(
            num_embeddings=n_codes,
            embedding_dim=n_code_dim,
            commitment_cost=commitment_cost,
        )

        # Combine positional embedding
        self.emb = self.vq._embedding
        self.pos_emb = nn.Embedding(n_pos, n_code_dim)
        create_position_codes(n_pos, n_code_dim, out=self.pos_emb.weight)

        block_config = GPTConfig(
            vocab_size=n_codes,
            n_embd=n_code_dim,
            n_head=n_head,
            n_layer=n_layer,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            block_size=n_pos,
        )
        self.transformer = nn.Sequential(*[Block(block_config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_code_dim)
        self.head = nn.Linear(n_code_dim, 1, bias=False)

        self.learning_rate = learning_rate

        self.input_to_label = ClassificationLabelTransform(
            ratio=1, sp1_idx=sp1_idx, sp2_idx=sp2_idx, pad_idx=pad_idx, unigram=False
        )

    def encode(self, input_ids, xa):
        xw = self.word_embedding(input_ids)
        xw = self.modal_act(self.word_projection(xw))

        reshape = False
        if xa.ndim == 4:
            B, N, n_channel, num_frames = xa.shape
            xa = xa.contiguous().view(-1, n_channel, num_frames)
            reshape = True

        xa = self.acoustic_encoder(xa)
        xa = xa.flatten(1)
        xa = self.modal_act(self.acoustic_projection(xa))
        if reshape:
            xa = xa.view(B, N, -1)

        x = torch.cat((xw, xa), dim=-1)
        x = self.pre_vq(x)
        emb, enc, vq_loss = self.vq(x)
        return emb, enc, vq_loss

    def pre_transformer_embedding(self, emb):
        # positional embedding for transformer
        B, N, _ = emb.size()
        positions = torch.arange(N).repeat(B, 1).to(emb.device)
        pos_emb = self.pos_emb(positions)

        # total embedding
        emb = emb + pos_emb
        return emb

    def forward(self, input_ids, xa):
        emb, enc, vq_loss = self.encode(input_ids, xa)
        emb = self.pre_transformer_embedding(emb)
        h = self.transformer(emb)
        h = self.ln_f(h)
        y_pred = self.head(h)
        return y_pred, vq_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def load_word_embeddings(self, state_dict_path, trainable=True):
        state_dict = torch.load(state_dict_path)
        self.word_embedding.load_state_dict(state_dict)
        if trainable:
            self.word_embedding.weight.requires_grad = False

    def shared_step(self, batch):
        x = []
        if "f0" in batch:
            x.append(batch["f0"])

        if "rms" in batch:
            x.append(batch["rms"])
        x = torch.stack(x, dim=-2)
        x, y, inp = self.input_to_label(x, batch["input_ids"])
        if x.ndim == 2:
            x = x.squeeze(-2)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)
        return x, y, inp

    def loss_function(self, y_pred, y):
        return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y.float())

    def training_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)

        y_pred, vq_loss = self(input_ids, x)
        loss = self.loss_function(y_pred, y)
        self.log("ploss", loss)
        loss += vq_loss

        self.log("loss", loss)
        self.log("vq_loss", vq_loss)
        return {"loss": loss, "vq_loss": vq_loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred, vq_loss = self(input_ids, x)
        loss = self.loss_function(y_pred, y)
        self.log("val_ploss", loss)
        loss += vq_loss
        self.log("val_loss", loss)
        self.log("val_vq_loss", vq_loss)
        return {"val_loss": loss, "val_vq_loss": vq_loss}

    def test_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred, vq_loss = self(input_ids, x)
        loss = self.loss_function(y_pred, y)
        loss += vq_loss
        # if n_batch == 0:
        #     emb, enc, _ = self.encode(input_ids, x)
        #     self.plot_probs_and_most_likely_code()

        return {"test_loss": loss, "test_vq_loss": vq_loss}

    def qualitative_plot(self, batch, tokenizer, max_samples=4, plot=False):
        with torch.no_grad():
            x, y, input_ids = self.shared_step(batch)
            y_pred, vq_loss = self(input_ids.to(self.device), x.to(self.device))
            emb, enc, _ = self.encode(input_ids.to(self.device), x.to(self.device))
            loss = self.loss_function(y_pred, y.to(self.device))
            loss += vq_loss
            probs = torch.sigmoid(y_pred).cpu()

        tokens = convert_ids_to_tokens(input_ids, tokenizer)
        sp = input_ids == self.sp1_idx
        sp += input_ids == self.sp2_idx
        targets = sp[:, 1:].cpu()

        n_plots = min(input_ids.shape[0], max_samples)
        fig, ax = plt.subplots(n_plots, 1, figsize=(16, 3 * n_plots))
        for i in range(input_ids.shape[0]):
            if i == max_samples:
                break
            N = len(probs[i])
            X = torch.arange(N)
            ax[i].bar(X, probs[i], alpha=0.6)
            ax[i].vlines(
                torch.where(targets[i])[0], ymin=0, ymax=1, alpha=0.1, color="g"
            )
            ax[i].set_ylim([0, 1])
            ax[i].set_xlim([0, N])
            ax[i].set_xticks(X)
            ax[i].set_xticklabels(tokens[i], rotation=65, fontsize=8)
            code = [str(d) for d in enc[0].argmax(dim=-1).tolist()]
            for j in range(N):
                yy = (7 - j % 10) * 0.05 + 0.35
                ax[i].text(x=j - 0.5, y=yy, s=code[j], fontsize=8)
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    def projection_plot(self):
        projection = umap.UMAP(
            n_neighbors=3, min_dist=0.1, metric="cosine"
        ).fit_transform(self.vq._embedding.weight.data.cpu())
        fig, ax = self.plot_codes(projection, plot=False)
        return fig, ax, projection

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # acoustic
        parser.add_argument("--prosody_frames", type=int, default=100)
        parser.add_argument("--prosody_in_channels", type=int, default=2)
        parser.add_argument("--prosody_conv_hidden", type=int, default=32)
        parser.add_argument("--prosody_conv_layers", type=int, default=5)
        parser.add_argument("--prosody_activation", type=str, default="ReLU")
        parser.add_argument(
            "--prosody_kernel",
            nargs="*",
            type=int,
            default=[5, 5, 5, 3, 3, 3],
        )
        parser.add_argument(
            "--prosody_stride",
            nargs="*",
            type=int,
            default=[2, 2, 2, 1, 1, 1],
        )
        parser.add_argument(
            "--prosody_padding",
            nargs="*",
            type=int,
            default=0,
        )

        # words
        parser.add_argument("--n_tokens", type=int, default=50259)
        parser.add_argument("--n_token_dim", type=int, default=768)
        parser.add_argument("--n_layer", type=int, default=1)
        parser.add_argument("--n_head", type=int, default=2)

        # embedding
        parser.add_argument("--n_codes", type=int, default=256)
        parser.add_argument("--n_code_dim", type=int, default=300)

        # Training and Data
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        return parser


def PGPTExperiment(parser):
    from turngpt.acousticDM import AudioDM

    parser = ProsodyGPT.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        # datasets=["switchboard"],
        f0=True,
        waveform=False,
        f0_normalize=True,
        f0_interpolate=True,
        f0_smooth=False,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    args.early_stopping = True

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    name = "PGPT"
    for n_code_dim in [100]:  # [300, 100, 60]:
        args.n_code_dim = n_code_dim
        for n_layer in [1]:
            args.n_layer = n_layer
            for n_head in [2]:
                args.n_head = n_head
                for n_codes in [10, 20, 40, 80, 100, 150, 200]:
                    args.n_codes = n_codes
                    model = ProsodyGPT(
                        n_codes=args.n_codes,
                        n_code_dim=args.n_code_dim,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        n_tokens=args.n_tokens,
                        n_token_dim=args.n_token_dim,
                        prosody_frames=args.prosody_frames,
                        prosody_in_channels=args.prosody_in_channels,
                        prosody_conv_hidden=args.prosody_conv_hidden,
                        prosody_kernel=args.prosody_kernel,
                        prosody_stride=args.prosody_stride,
                        prosody_activation=args.prosody_activation,
                    )
                    model.load_word_embeddings(EMB_PATH)
                    n_params = get_n_trainable_params(model)
                    # print(model)
                    print("n_codes: ", args.n_codes)
                    print("n_code_dim: ", args.n_code_dim)
                    print("n_layer: ", args.n_layer)
                    print("n_head: ", args.n_head)
                    print("parameters: ", n_params)

                    logger, checkpoint_callback, callbacks = logging(args, name=name)

                    trainer = pl.Trainer.from_argparse_args(
                        args,
                        logger=logger,
                        checkpoint_callback=checkpoint_callback,
                        callbacks=callbacks,
                    )
                    trainer.fit(model, datamodule=dm)
                    trainer.test(test_dataloaders=dm.val_dataloader(), verbose=False)

                    # plots
                    batch = next(iter(dm.val_dataloader()))
                    fig, ax = model.qualitative_plot(batch, dm.tokenizer, max_samples=4)
                    fig_path = join(model.logger.log_dir, "sample")
                    fig.savefig(fig_path)
                    print("saved plot -> ", fig_path)

                    fig, ax, proj = model.projection_plot()
                    fig_path = join(model.logger.log_dir, "projection")
                    fig.savefig(fig_path)
                    print("saved plot -> ", fig_path)


def evaluate():
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser = ProsodyGPT.add_model_specific_args(parser)
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

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))

    checkpoint = "checkpoints/PGPT/version_0/checkpoints/epoch=10-val_loss=0.13770.ckpt"
    hparams = "checkpoints/PGPT/version_0/hparams.yaml"
    # model = ProsodyGPT.load_from_checkpoint(checkpoint, hparams_file=hparams)
    model = ProsodyGPT.load_from_checkpoint(checkpoint)
    model.eval()

    batch = next(iter(dm.val_dataloader()))
    fig, ax = model.qualitative_plot(batch, dm.tokenizer, plot=True, max_samples=2)
    fig, ax, proj = model.projection_plot()
    plt.pause(0.01)


if __name__ == "__main__":

    from matplotlib import use as mpl_use

    mpl_use("Agg")
    parser = ArgumentParser()
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    PGPTExperiment(parser)

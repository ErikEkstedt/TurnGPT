from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turngpt.models.gpt_mini import GPT


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VectorQuantizerEMA(nn.Module):
    """
    Slightly changed version of Zalando research vq-vae implementation

    Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
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


class VQEncDec(pl.LightningModule):
    def __init__(
        self, input_size=80, hidden=32, n_codes=64, code_dim=32, learning_rate=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.codebook = VectorQuantizerEMA(
            num_embeddings=n_codes, embedding_dim=code_dim, commitment_cost=0.2, decay=0
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, input_size)
        )

    def forward(self, x):
        """
        x: (B, N, T, d)
        """
        B, N, T, d = x.shape
        z = self.encoder(x.flatten(-2))
        codes, codes_oh, loss = self.codebook(z)
        x_rec = self.decoder(codes)
        x_rec = x_rec.view(B, N, T, d)
        return {
            "z": z,
            "codes": codes,
            "x_rec": x_rec,
            "codes_oh": codes_oh,
            "vq_loss": loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def loss_function(self, x_rec, x, vq_loss):
        recon_error = F.mse_loss(x_rec, x)
        loss = recon_error + vq_loss

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        return loss

    def shared_step(self, batch):
        input_ids, _, _, spf = batch
        return spf

    def training_step(self, batch, *args, **kwargs):
        x = self.shared_step(batch)
        out = self(x)
        loss = self.loss_function(out["x_rec"], x, out["vq_loss"])
        self.log(
            "vq_loss",
            out["vq_loss"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = self.shared_step(batch)
        out = self(x)
        loss = self.loss_function(out["x_rec"], x, out["vq_loss"])

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_size", default=80)
        parser.add_argument("--hidden", default=32)
        parser.add_argument("--n_codes", default=64)
        parser.add_argument("--code_dim", default=32)

        parser.add_argument("--learning_rate", default=1e-4)
        return parser


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from acousticgpt.acousticDM import AcousticGPTDM
    from ttd.utils import get_run_dir
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from os.path import join
    from os import environ

    parser = ArgumentParser()
    parser = AcousticGPTDM.add_data_specific_args(parser, datasets=["maptask"])
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQEncDec.add_model_specific_args(parser)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    args.prosody = True

    dm = AcousticGPTDM(args)
    dm.prepare_data()
    dm.setup("fit")

    model = VQEncDec(
        input_size=args.input_size,
        hidden=args.hidden,
        n_codes=args.n_codes,
        code_dim=args.code_dim,
    )

    # Where to save the training
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print("root dir: ", args.save_dir)
    checkpoint_callback = None
    callbacks = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        name = "VQEncDec"
        desc = f"{name} training"
        logger = TensorBoardLogger(args.save_dir, name=name)
        ch_path = join(logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}-{val_loss:.5f}",
            save_top_k=2,
            mode="min",
            monitor="val_loss",
        )

        # Save the used tokenizer
        tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        torch.save(dm.tokenizer, tokenizer_path)
        print("tokenizer saved -> ", tokenizer_path)

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

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=dm)

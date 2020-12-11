from argparse import ArgumentParser
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import umap

from turngpt.prosody_classifier import ConvEncoder, logging, VectorQuantizerEMA
from turngpt.transforms import ClassificationLabelTransform

import matplotlib.pyplot as plt


def get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx):
    inp = input_ids.clone()
    inp[inp == sp2_idx] = sp1_idx
    sp_b, sp_inds = torch.where(inp == sp1_idx)  # all speaker 1 tokens
    return (sp_b, sp_inds)


def get_turn_shift_indices(input_ids, sp1_idx, sp2_idx):
    ts_bs, ts_inds = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)
    ts_inds = ts_inds - 1  # turn-shift are
    ts_bs = ts_bs[ts_inds != -1]
    ts_inds = ts_inds[ts_inds != -1]
    return (ts_bs, ts_inds)


def get_positive_and_negative_indices(input_ids, sp1_idx, sp2_idx, pad_idx):
    """
    Finds positive and negative indices for turn-shifts.

    * Positive turn-shifts are the indices prior to a <speaker1/2> token
    * Negative turn-shifts are all other indices (except pad_tokens)

    Returns:
        turn_shift_indices:     tuple, (batch, inds) e.g.  input_ids[turn_shift_indices]
        non_turn_shift_indices: tuple, (batch, inds) e.g.  input_ids[non_turn_shift_indices]
    """
    (ts_bs, ts_inds) = get_turn_shift_indices(input_ids, sp1_idx, sp2_idx)
    bp, indp = torch.where(input_ids != pad_idx)  # all valid places

    # TODO:
    # Remove the speaker-id tokens from negatives?

    neg_bs, neg_inds = [], []
    for i in bp.unique():
        neg_ind = indp[bp == i]  # valid indices (not pad) # [1:]  # omit 0
        ts = ts_inds[ts_bs == i]  # turn-shifts in batch i
        neg_ind[ts] = -1  # mark these
        neg_ind = neg_ind[neg_ind != -1]
        neg_bs.append(torch.ones_like(neg_ind) * i)
        neg_inds.append(neg_ind)

    neg_bs = torch.cat(neg_bs)
    neg_inds = torch.cat(neg_inds)
    return (ts_bs, ts_inds), (neg_bs, neg_inds)


LM_PATH = "checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981.ckpt"
LM_HPARAMS = "checkpoints/pretrained/PDECTWW/hparams.yaml"
EMB_PATH = "checkpoints/pretrained/PDECTWW/word_embedding_state_dict.pt"


def load_embeddings():
    state_dict = torch.load(EMB_PATH)
    shape = num_embeddings = state_dict["weight"].shape
    emb = nn.Embedding(num_embeddings=shape[0], embedding_dim=shape[1])
    emb.load_state_dict(state_dict)
    emb.weight.requires_grad = False
    return emb


def save_embedding_state_dict():
    from turngpt.TurnGPT import TurnGPT

    emb = load_embeddings()
    state_dict = emb.state_dict()
    torch.save(state_dict, EMB_PATH)
    print("saved word embedding -> ", EMB_PATH)


###########################################################################
class ProsodyWordClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=2,
        conv_hidden=64,
        layers=5,
        kernel=3,
        first_stride=2,
        fc_hidden=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50259,
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.sp1_idx = sp1_idx
        # self.sp2_idx = sp2_idx
        # self.pad_idx = pad_idx

        # Words
        self.word_embedding = load_embeddings()
        self.w2h = nn.Linear(self.word_embedding.embedding_dim, fc_hidden)

        # Prosody
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            # num_layers=layers,
            # kernel_size=kernel,
            # first_stride=first_stride,
            num_layers=7,
            kernel_size=3,
            first_stride=[2, 2, 1, 1, 1, 1, 1],
            activation=activation,
        )
        fc_in = int(self.conv_encoder.output_size * conv_hidden)
        self.p2h = nn.Linear(fc_in, fc_hidden)
        self.head = nn.Linear(fc_hidden * 2, 1)

        self.learning_rate = learning_rate

        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, input_ids, x):
        xw = self.word_embedding(input_ids)
        xw = torch.sigmoid(self.w2h(xw))
        x = self.conv_encoder(x)
        x = torch.sigmoid(self.p2h(x.flatten(1)))
        x = self.head(torch.cat((xw, x), dim=-1))
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

        y_pred = self(input_ids, x)
        loss = self.loss_function(y_pred, y)

        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, *args, **kwargs):
        x, y, input_ids = self.shared_step(batch)
        y_pred = self(input_ids, x)
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


def PWexperiment(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """
    from turngpt.acousticDM import AudioDM

    parser = ProsodyWordClassifier.add_model_specific_args(parser)
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

    args.prosody_conv_hidden = 64

    for hidden in [32, 64, 96]:
        args.prosody_fc_hidden = hidden

        name = "PW"
        in_channels = 0
        if args.rms:
            in_channels += 1
            name += "_rms"
        if args.f0:
            in_channels += 1
            name += "_f0"

        print(name)
        print("conv hidden: ", args.prosody_conv_hidden)
        print("fc hidden: ", args.prosody_fc_hidden)
        model = ProsodyWordClassifier(
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
        trainer.test(test_dataloaders=dm.val_dataloader())


###########################################################################
class ProsodyWordQClassifier(pl.LightningModule):
    def __init__(
        self,
        in_frames=100,
        in_channels=2,
        conv_hidden=64,
        layers=5,
        kernel=3,
        n_codes=64,
        n_code_dim=32,
        first_stride=2,
        fc_hidden=32,
        learning_rate=1e-3,
        activation="ReLU",
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50259,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Words
        self.word_embedding = load_embeddings()
        self.w2h = nn.Linear(self.word_embedding.embedding_dim, fc_hidden)

        # Prosody
        self.conv_encoder = ConvEncoder(
            input_frames=in_frames,
            hidden=conv_hidden,
            in_channels=in_channels,
            # num_layers=layers,
            # kernel_size=kernel,
            # first_stride=first_stride,
            num_layers=7,
            kernel_size=3,
            first_stride=[2, 2, 1, 1, 1, 1, 1],
            activation=activation,
        )
        fc_in = int(self.conv_encoder.output_size * conv_hidden)
        self.p2h = nn.Linear(fc_in, fc_hidden)

        # How to make the codes sparse?
        self.pre_vq = nn.Linear(fc_hidden * 2, n_code_dim)
        self.vq = VectorQuantizerEMA(num_embeddings=n_codes, embedding_dim=n_code_dim)

        self.head = nn.Linear(n_code_dim, 1)

        self.learning_rate = learning_rate

        self.input_to_label = ClassificationLabelTransform(
            ratio=1,
            sp1_idx=sp1_idx,
            sp2_idx=sp2_idx,
            pad_idx=pad_idx,
            max_batch_size=512,
        )

    def forward(self, input_ids, x):
        xw = self.word_embedding(input_ids)
        xw = torch.sigmoid(self.w2h(xw))
        x = self.conv_encoder(x)
        x = torch.sigmoid(self.p2h(x.flatten(1)))
        x = self.pre_vq(torch.cat((xw, x), dim=-1).unsqueeze(1))
        z_q, enc, vq_loss = self.vq(x)
        x = self.head(z_q.squeeze(1))
        return x, vq_loss

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


def PWQexperiment(parser):
    """
    Simple prosodic classfier:
        Conv1d:    downsample ~2, kernel_size, layers, hidden
        fc:         one hidden layer decoder.  conv_hidden  -> 32 -> 1

    Classification of pre turn-shift words vs all other words. BCELoss (with logits)
    """
    from turngpt.acousticDM import AudioDM

    parser = ProsodyWordQClassifier.add_model_specific_args(parser)
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
    args.learning_rate = 1e-2

    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")

    args.prosody_n_code_dim = 64
    args.prosody_conv_hidden = 64
    for layers in [5, 7, 9]:
        args.prosody_conv_layers = layers
        for n_codes in [32, 64, 128]:
            args.prosody_n_codes = n_codes

            name = "PW"
            in_channels = 0
            if args.rms:
                in_channels += 1
                name += "_rms"
            if args.f0:
                in_channels += 1
                name += "_f0"

            print(name)
            print("conv hidden: ", args.prosody_conv_hidden)
            print("conv layers: ", args.prosody_conv_layers)
            print("conv n_codes: ", args.prosody_n_codes)
            print("conv code_dim: ", args.prosody_n_code_dim)

            model = ProsodyWordQClassifier(
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
            trainer.test(test_dataloaders=dm.val_dataloader)


def debug():
    from turngpt.acousticDM import AudioDM

    parser = ArgumentParser()
    parser = ProsodyWordClassifier.add_model_specific_args(parser)
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

    model = ProsodyWordClassifier()

    o = model.training_step(batch)


if __name__ == "__main__":

    from matplotlib import use as mpl_use

    mpl_use("Agg")
    pl.seed_everything(1234)

    parser = ArgumentParser()
    PWexperiment(parser)
    PWQexperiment(parser)

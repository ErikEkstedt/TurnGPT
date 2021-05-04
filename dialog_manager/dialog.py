from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from turngpt.models.gpt_mini import GPT
from turngpt.models import Attention1D


class DialogManager(pl.LightningModule):
    """
    ReImplementation or based on of Rasas "Dialog Transformer": https://arxiv.org/pdf/1911.00486.pdf
    """

    def __init__(
        self,
        code_dim,
        n_intents=1,
        n_slots=1,
        n_actions=2,
        # n_vocab,
        dialog_hidden=256,
        dialog_n_head=8,
        dialog_n_layer=2,
        dialog_embd_pdrop=0.1,
        dialog_resid_pdrop=0.1,
        dialog_attn_pdrop=0.1,
        dialog_use_speaker_emb=False,
        dialog_context=128,
        # mixer_n_head=4,
        intent_to_idx={"greeting": 0, "affirm": 1},
        slot_to_idx={"name": 0},
        action_to_idx={"greeting": 0, "affirm": 1},
        learning_rate=1e-4,
        weight_decay=0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        # parameters
        self.code_dim = code_dim
        self.n_intents = (
            n_intents + 1
        )  # adding the "none" token or complete omission of intent
        self.n_slots = n_slots + 1
        self.n_actions = n_actions + 1
        self.n_total_codebook_size = n_intents + n_slots
        # self.n_vocab = n_vocab

        # helpers
        self.intent_to_idx = intent_to_idx
        self.slot_to_idx = slot_to_idx
        self.action_to_idx = action_to_idx

        # NN
        self.codebook_intents = nn.Embeding(n_intents, n_intents)
        self.codebook_slots = nn.Embeding(n_slots, n_slots)
        self.codebook_actions = nn.Embeding(n_slots, n_actions)

        # self.codebook_mixer = Attention1D(
        #     code_dim, features_in=3, features_out=1, num_heads=mixer_n_head
        # )
        self.dialog_transformer = GPT(
            n_vocab=self.n_total_codebook_size,
            n_embd=dialog_hidden,
            n_head=dialog_n_head,
            n_layer=dialog_n_layer,
            embd_pdrop=dialog_embd_pdrop,
            resid_pdrop=dialog_resid_pdrop,
            attn_pdrop=dialog_attn_pdrop,
            use_speaker_emb=dialog_use_speaker_emb,
            chunk_size=dialog_context,
        )

        # Training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def embedding(self, x_intent, x_slot, x_action):
        code_intent = self.codebook_intents(x_intent)
        code_slot = self.codebook_intents(x_slot)
        code_action = self.codebook_intents(x_action)
        total_code = code_intent + code_slot + code_action
        # return torch.stack((code_intent, code_slot, code_action), dim=-1)
        return total_code

    def forward(self, x_intent, x_slot, x_action, y_action):
        codes = self.embedding(x_intent, x_slot, x_action)
        out = self.dialog_transformer(
            codes
        )  # out['logits']: B, N, self.n_total_codebook
        out["codes"] = codes

        out["label_codes"] = self.codebook_action(y_action)
        return out

    def similarity(self, h_action_pos, h_pos, h_neg):
        S_plus = torch.dot(h_action_pos, h_pos)
        S_neg = torch.dot(h_action_pos, h_neg)
        return S_plus, S_neg

    def get_negative_samples(self, h):
        B, N, T = h.size()

        neg_codes = torch.arange(B*N)

        return self.transformer.head(h)

    def loss_function_similarity(self, h_hat, h):

        h_neg = self.get_negative_samples(h_hat)
        S_plus, S_neg = self.similarity(h_hat, h, h_neg)
        loss = 

        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def string_to_intent(self, intent):
        return self.intent_to_idx(intent)

    def string_to_slot(self, intent):
        return self.intent_to_idx(intent)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--n_slots", default=4, type=int)
        parser.add_argument("--n_intents", default=4, type=int)
        parser.add_argument("--code_dim", default=64, type=int)

        parser.add_argument("--dialog_n_head", type=int, default=8)
        parser.add_argument("--dialog_n_layer", type=int, default=2)
        parser.add_argument("--dialog_embd_pdrop", type=float, default=0.1)
        parser.add_argument("--dialog_resid_pdrop", type=float, default=0.1)
        parser.add_argument("--dialog_attn_pdrop", type=float, default=0.1)
        parser.add_argument("--dialog_use_speaker_emb", action="store_true")
        parser.add_argument("--dialog_chunk_size", type=int, default=128)

        # Training
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--weight_decay", default=1e-3, type=float)
        return parser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = DialogManager.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model = DialogManager(
        code_dim=args.code_dim,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from turngpt.pl_modules import TurnGPT

from transformers import GPT2LMHeadModel, GPT2Config, AdamW


class TurnGPTModel(nn.Module):
    def __init__(self, n_vocab=None, dropout=0.1, pretrained="gpt2"):
        super().__init__()

        if isinstance(pretrained, str):
            if pretrained.lower() == "none":
                pretrained = None

        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
        )
        self.config = self.model.config
        self.block_size = self.config.n_positions

        if n_vocab is not None:
            self.extend_embeddings(n_vocab)

    def load_model(self, checkpoint_path=None, tokenizer_path=None, tokenizer=None):
        if tokenizer is None:
            assert (
                tokenizer_path is not None
            ), "Must provide either a tokenizer or a path to load it"
            tokenizer = torch.load(tokenizer_path)

        print(f"Loading TurnGPT...")
        self.extend_embeddings(len(tokenizer))

        try:
            checkpoint = torch.load(checkpoint_path)
            keys = self.model.load_state_dict(checkpoint["model"], strict=False)
        except:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu"), strict=False
            )
            keys = self.model.load_state_dict(checkpoint["model"])
        print("Loaded PreTrained Weights")
        print("missing keys: ", keys.missing_keys)
        print("unexpected keys: ", keys.unexpected_keys)
        return tokenizer

    def extend_embeddings(self, tokenizer_len):
        """
        resize_token_embeddings expect to receive the full size of the new vocabulary,
        i.e. the length of the tokenizer_len.
        """
        # wte_size = self.model.get_embedding_size()
        wte_size = self.model.transformer.wte.weight.shape[0]
        # Check if resize is needed
        if tokenizer_len != wte_size:
            print("Extending vocabulary")
            print("Append", tokenizer_len - wte_size, "tokens")
            self.model.resize_token_embeddings(
                tokenizer_len
            )  # ties weights and extend self.model.lm_head to match
            print("Resized model embedding -> ", tokenizer_len)

    def body(
        self,
        input_ids,
        speaker_ids,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
    ):
        # transformer_outputs: last hidden state, (presents), (all hidden_states), (attentions)
        transformer_outputs = self.model.transformer(
            input_ids,
            token_type_ids=speaker_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        outputs = {"z": transformer_outputs[0]}
        i = 1
        if use_cache:
            outputs["presents"] = transformer_outputs[i]
            i += 1

        if output_hidden_states:
            outputs["hidden"] = torch.stack(transformer_outputs[i], dim=1)
            i += 1

        if output_attentions:
            outputs["attention"] = torch.stack(transformer_outputs[i], dim=1)
        return outputs

    def body_from_embedding(self, inputs_embeds, speaker_ids):
        input_shape = inputs_embeds.size()[:-1]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # head_mask = self.model.get_head_mask(None, self.model.config.n_layer)
        # print("head mask: ", head_mask)

        # Position Embeddings
        position_ids = torch.arange(
            input_shape[-1], dtype=torch.long, device=inputs_embeds.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds = self.model.transformer.wpe(position_ids)

        if speaker_ids is not None:
            token_type_embeds = self.model.transformer.wte(speaker_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.model.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.model.transformer.h):
            outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=None,
            )
            hidden_states = outputs[0]

        hidden_states = self.model.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        return {"z": hidden_states}

    def forward(
        self,
        input_ids,
        speaker_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
    ):
        """
        TurnGPT forward pass

        :param input_ids:               Torch.LongTensor of input indices
        :param speaker_ids:             Torch.LongTensor, (Optional), Torch.LongTensor of speaker indices
        :param output_attentions:       Bool, (Default=False)
        :param output_hidden_states:    Bool, (Default=False)
        :param use_cache:               Bool, (Default=False)

        """
        # transformer_outputs: last hidden state, (presents), (all hidden_states), (attentions)
        outputs = self.body(
            input_ids,
            speaker_ids=speaker_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        outputs["logits"] = self.model.lm_head(outputs["z"])
        return outputs


class TurnGPTPretrained(TurnGPT):
    def __init__(self, n_vocab, pad_idx, dropout=0.1, pretrained="gpt2", **kwargs):
        super().__init__()
        self.pad_idx = pad_idx

        self.model = TurnGPTModel(
            n_vocab=n_vocab,
            dropout=dropout,
            pretrained=pretrained,
        )

        self.chunk_size = self.model.config.n_positions
        self.n_embd = self.model.config.n_embd
        self.n_head = self.model.config.n_head
        self.n_layer = self.model.config.n_layer

        # save
        self.save_hyperparameters()

    def configure_optimizers(self):
        return AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate, correct_bias=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model
        parser.add_argument("--pretrained", type=str, default="gpt2")
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--chunk_size", type=int, default=512)

        # Training
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser

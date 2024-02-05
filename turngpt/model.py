from argparse import ArgumentParser

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

import wandb
from turngpt.generation import generate
from turngpt.plot_utils import plot_trp
from turngpt.projection_labeler import ProjectionLabeler
from turngpt.tokenizer import SpokenDialogTokenizer


def load_transformer(
    pretrained_model_name_or_path="gpt2", pretrained=True, **model_kwargs
):
    """Load transformer model. If `pretrained` then initialize with pretrained weights otherwise start from scratch"""

    implemented = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
    ]

    update_on_pretrain = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]

    if not (
        "gpt2" in pretrained_model_name_or_path.lower()
        or "dialogpt" in pretrained_model_name_or_path.lower()
    ):
        raise NotImplementedError(
            f"pretrained_model_name_or_path=`{pretrained_model_name_or_path}` is Not implemented! Please use: {implemented}"
        )

    config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
    if pretrained:
        # Update only certain parameters
        for k, v in model_kwargs.items():
            if k in update_on_pretrain and v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, config=config
        )
    else:
        for k, v in model_kwargs.items():
            if v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel(config=config)
    return transformer


class Utils:
    tokenizer: SpokenDialogTokenizer

    def idx_to_string(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        s = self.tokenizer.convert_ids_to_tokens(idx)
        s = self.tokenizer.convert_tokens_to_string(
            s.strip()
        )  # remove prefix space/symbol
        return s

    def get_trp(self, x):
        return x[..., self.tokenizer.eos_token_id]

    def tokenize_strings(self, string_or_list, add_post_eos_token=False):
        if isinstance(string_or_list, str) and add_post_eos_token:
            if not string_or_list.strip().endswith(self.tokenizer.eos_token):
                string_or_list += self.tokenizer.eos_token

        t = self.tokenizer(string_or_list, return_tensors="pt")
        if not isinstance(t["input_ids"], torch.Tensor):
            tmp_inp, tmp_sp = [], []
            for inp, sp in zip(t["input_ids"], t["speaker_ids"]):
                tmp_inp.append(torch.tensor(inp))
                tmp_sp.append(torch.tensor(sp))
            tmp = self.tokenizer.pad({"input_ids": tmp_inp})
            t["input_ids"] = tmp["input_ids"]
            t["attention_mask"] = tmp["attention_mask"]
            t["speaker_ids"] = self.tokenizer.pad({"input_ids": tmp_sp})["input_ids"]

        for k, v in t.items():
            t[k] = v.to(self.device)
        return t

    def get_tokens(self, input_ids):
        def inner(input_ids):
            # inner_tokens = []
            # for idx in input_ids:
            #     inner_tokens.append(self.idx_to_string(idx))
            # return inner_tokens
            return self.idx_to_string(input_ids)

        def outer(input_ids):
            tokens = []
            for batch in input_ids:
                tokens.append(inner(batch))
            return tokens

        tokens = None
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim > 2:
                raise LookupError(
                    f"self.get_tokens not implemented for Tensor shape: {tuple(input_ids.shape)} (>2)"
                )
            elif input_ids.ndim == 2:
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
                for idx in input_ids:
                    tokens.append(self.idx_to_string(idx))
        elif isinstance(input_ids, list):
            if isinstance(input_ids[0], list):
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
        return tokens

    @torch.no_grad()
    def string_list_to_trp(
        self, string_or_list, add_post_eos_token=False, **model_kwargs
    ):
        t = self.tokenize_strings(string_or_list, add_post_eos_token=add_post_eos_token)

        # Model
        out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        out["tokens"] = self.get_tokens(t["input_ids"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()
        return out


class TurnGPTWandbCallbacks(Callback):
    turn_list = [
        ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
        [
            "Hello there I basically had the worst day of my life",
            "Oh no, what happened?",
            "Do you want the long or the short story?",
        ],
    ]

    def __init__(
        self,
        text_list=None,
        n_steps=200,
        n_generate=20,
        eos_token="<ts>",
        unk_token="<|endoftext|>",
    ):
        super().__init__()
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.text_list = text_list
        self.n_steps = n_steps
        self.n_generate = n_generate
        if self.text_list is None:
            self.text_list = self.turn_list

    def trp_plots(self, trainer, pl_module, name="TRP/example"):
        out = pl_module.string_list_to_trp(self.text_list)

        for b in range(out["trp_probs"].shape[0]):
            proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
            fig, _ = plot_trp(
                trp=out["trp_probs"][b].cpu(),
                proj=proj,
                text=out["tokens"][b],
                unk_token=pl_module.tokenizer.unk_token,
                eos_token=pl_module.tokenizer.eos_token,
                plot=False,
            )

            pl_module.logger.experiment.log(
                data={
                    f"{name}_{b}": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            plt.close("all")

    def generate(self, trainer, pl_module, name):
        gen = generate(
            pl_module,
            context=self.turn_list[-1],
            n_steps=self.n_steps,
            top_p=0.9,
            top_k=-1,
            n_trajectories=self.n_generate,
            strategy="sample",
            stop_at_eos=True,
        )
        # remove duplicates
        l = (gen["input_ids"][0] != -1).sum()
        G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
        for i, g in enumerate(gen["tokens"][1:]):
            if g not in G["tokens"]:
                l = (gen["input_ids"][i] != -1).sum()
                G["tokens"].append(g)
                G["probs"].append(gen["probs"][i][:l].cpu())

        table = wandb.Table(
            columns=["context", "sample", "probs"],
            data=[
                ["... " + self.turn_list[-1][-1], toks, probs.tolist()]
                for toks, probs in zip(G["tokens"], G["probs"])
            ],
        )
        pl_module.logger.experiment.log(
            data={
                f"{name}": table,
                "global_step": trainer.global_step,
            },
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.trp_plots(trainer, pl_module, name="TRP/example")
        self.generate(trainer, pl_module, name="Gen")

    def on_save_checkpoint(self, trainer, pl_module, *args, **kwargs):
        self.trp_plots(trainer, pl_module, name="TRP-chpt/example")
        self.generate(trainer, pl_module, name="Gen-chpt")


class TurnGPT(L.LightningModule, Utils):
    """
    This is the code example of teaching and research.

    Add features to this model i.e. analysis of turn-taking.


    On training
    * call `model.initialize_special_embeddings()` to initialize <ts> = eos_token
    """

    def __init__(
        self,
        pretrained_model_name_or_path="gpt2",
        pretrained=True,
        trp_projection_steps=-1,
        trp_projection_type="linear",
        omit_dialog_states=False,
        no_train_first_n=5,
        learning_rate=1e-4,
        weight_loss=False,
        weight_regular_token=0.5,
        weight_eos_token=1.0,
        **model_kwargs,
    ):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        self.pretrained = pretrained

        # train parameters
        self.no_train_first_n = no_train_first_n
        self.learning_rate = learning_rate
        self.weight_loss = weight_loss
        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.omit_dialog_states = omit_dialog_states

        # Load `transformers` model
        self.transformer = load_transformer(
            pretrained_model_name_or_path, pretrained=pretrained, **model_kwargs
        )

        # TRP projection head
        self.trp_projection_steps = trp_projection_steps
        if trp_projection_steps > 0:
            self.trp_projection_type = trp_projection_type
            hidden_size = self.transformer.config.hidden_size

            # MultiTask Head operating on n last hidden states
            if trp_projection_type.lower() == "attention":
                raise NotImplementedError()
            else:
                self.trp_projection_head = nn.Linear(hidden_size, 1)

        self.save_hyperparameters()

    @property
    def run_name(self):
        name = "TurnGPT"
        if self.trp_projection_steps > 0:
            name += f"_proj_{self.trp_projection_steps}"
        return name

    def init_tokenizer(self):
        # The tokenizer should always be a part of the model
        self.tokenizer = SpokenDialogTokenizer(self.name_or_path)

        # Add extra embeddings for custom tokens
        # Optional: Initialize <ts> to be close to punctuation tokens.
        self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

    def initialize_special_embeddings(self, tokens=["!", "?", "."]):
        """
        Initialize `eos_token` as the average of `tokens`.

        By default (or looking at <speaker1/2>) the embeddings are initalized to m=0, std=0.02
        """
        ts = self.tokenizer.eos_token_id
        # pre = self.transformer.transformer.wte.weight[ts].clone()
        with torch.no_grad():
            ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            avg_emb = self.transformer.transformer.wte(ids).mean(0)
            self.transformer.transformer.wte.weight.data[ts] = avg_emb
        # post = self.transformer.transformer.wte.weight[ts]
        # print(pre == post)
        print(f"Initalized {self.tokenizer.eos_token} -> avg({tokens})")

    def print_parameters(self):
        print("")
        print("TurnGPT")
        print("name_or_path: ", self.name_or_path)
        print("learning_rate: ", self.learning_rate)
        print("weight_loss: ", self.weight_loss)
        if self.weight_loss:
            print("weight_regular_token: ", self.weight_regular_token)
            print("weight_eos_token: ", self.weight_eos_token)
        if self.trp_projection_steps > 0:
            print("trp_projection_steps: ", self.trp_projection_steps)
            print("trp_projection_type: ", self.trp_projection_type)
        print()

    def get_labels(self, input_ids, mask, value=-100):
        """Don't shift the labels (happens internally)"""
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = value

        if self.no_train_first_n > 0:
            labels[:, : self.no_train_first_n] = value
        return labels

    def get_projection_labels(self, input_ids, mask, value=-100):
        labeler = ProjectionLabeler(
            projection_steps=self.trp_projection_steps,
            token_id=self.tokenizer.eos_token_id,
        ).to(self.device)
        proj_labels = labeler(input_ids)
        proj_labels[torch.logical_not(mask)] = value
        if self.no_train_first_n > 0:
            proj_labels[:, : self.no_train_first_n] = value
        return proj_labels

    @torch.no_grad()
    def get_loss_weight(self):
        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float)
            * self.weight_regular_token
        )
        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        return weight.to(self.device)

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        """
        Taken from GPT2LMHeadModel in:

          https://github.com/huggingface/transformers/blob/91ff480e2693f36b11aaebc4e9cc79e4e3c049da/src/transformers/models/gpt2/modeling_gpt2.py#L968

        How to weight CE-Loss?

            https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10

        Using the custom weights gets a total loss that is larger than without?
        I don't want the model to train as much on the new type of text it receives but to learn the `eos_token` better.
        I assume that the loss should be less if I scale down the normal tokens loss...

        `CrossEntropyLoss` seems to normalize the loss if not `reduction=None` which account for the above behaviour.

        Instead we use `reduction=none` and simply average over the weighted loss values.
        Given that `weight_regular_token` < 1 and `weight_eos_token` >=1 we get a lower loss when weighting.

        Is this mathematically sound? well that is the question.

        I guess one could simply scale the `weight_eos_token > 1` while `weight_regular_token=1` and use a smaller learning rate?
        """
        weight = None
        if self.weight_loss:
            weight = self.get_loss_weight()

        loss_fct = nn.CrossEntropyLoss(weight=weight, reduction="none")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens and calc loss
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if reduction != "none":
            loss = loss.mean()
        return loss

    def bce_loss(self, logits, labels):
        """
        Simple BCELoss for binary trp projection

        Must extend this if multiple labels are to be used...
        """
        loss_fct = nn.BCEWithLogitsLoss()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1]  # , :].contiguous()
        shift_labels = labels[..., 1:]  # .contiguous()

        # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100
        loss = loss_fct(
            torch.masked_select(shift_logits, indices_for_training),
            torch.masked_select(shift_labels, indices_for_training),
        )
        # shift_logits = torch.masked_select(shift_logits, indices_for_training)
        # shift_labels = torch.masked_select(shift_labels, indices_for_training)
        # loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_likelihood(self, logits, labels, pad_last=True, pad_first=False):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_probs = shift_logits.softmax(dim=-1)

        # likelihood = shift_probs[shift_labels]
        bn = torch.ones_like(shift_labels)
        bn[0] = 0

        seq = torch.arange(shift_labels.shape[-1])
        seq = torch.stack([seq] * shift_labels.shape[0])

        likelihood = shift_probs[bn.view(-1), seq.view(-1), shift_labels.view(-1)]
        likelihood = likelihood.view(shift_labels.shape)
        if pad_first:
            likelihood = torch.cat(
                [torch.zeros(likelihood.shape[0], 1), likelihood], dim=-1
            )
        elif pad_last:
            likelihood = torch.cat(
                [likelihood, torch.zeros(likelihood.shape[0], 1)], dim=-1
            )
        return likelihood

    def forward(
        self,
        input_ids=None,
        speaker_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Simple rewrite of original:

            https://github.com/huggingface/transformers/blob/439a43b6b403205eeda2d62645fc16c93627d30d/src/transformers/models/gpt2/modeling_gpt2.py#L1086
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )

        transformer_outputs = self.transformer.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=speaker_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.transformer.model_parallel:
            torch.cuda.set_device(self.transformer.transformer.first_device)
            hidden_states = hidden_states.to(self.transformer.lm_head.weight.device)

        # Language Modeling
        lm_logits = self.transformer.lm_head(hidden_states)
        lm_loss = None
        if labels is not None:
            lm_loss = self.cross_entropy_loss(lm_logits, labels)

        # MultiTask Modeling
        mc_logits = None
        mc_loss = None
        if self.trp_projection_steps > 0:
            # NOTE:
            # Assumed to only guess a single class
            mc_logits = self.trp_projection_head(hidden_states).squeeze(-1)

            if mc_labels is not None:
                mc_loss = self.bce_loss(mc_logits, mc_labels)

        # if not return_dict:
        #     output = (lm_logits, mc_logits) + transformer_outputs[1:]
        #     if mc_loss is not None:
        #         output = (mc_loss,) + output
        #     return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def configure_optimizers(self):
        # NOTE:
        # Use multiple optimizers for transformer and projection?
        # see:
        #   https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#use-multiple-optimizers-like-gans-manual
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        """We must save the tokenizer used during training"""
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        """We must load the tokenizer used during training and resize the embeddings appropriately"""
        if "tokenizer" in checkpoint:
            print("#" * 70)
            print("LOAD CHECKPOINT TOKENIZER")
            self.tokenizer = checkpoint["tokenizer"]
            print("Loaded tokenizer")
            print(self.tokenizer)

            # Add extra embeddings for custom tokens
            self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
            print("Resized weights")
            print("#" * 70)

    def training_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )

        if self.trp_projection_steps > 0:
            self.log("loss_lm", out["loss"])
            self.log("loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            self.log("loss", out["loss"])
            total_loss = out["loss"]
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )

        if self.trp_projection_steps > 0:
            self.log("val_loss_lm", out["loss"])
            self.log("val_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            total_loss = out["loss"]

        self.log("val_loss", total_loss)


def test():
    checkpoint = "checkpoints/turngpt_epoch=11_val_loss=0.6381.ckpt"
    model = TurnGPT.load_from_checkpoint(checkpoint).eval()
    model.tokenizer = SpokenDialogTokenizer("gpt2")

    text_list = [
        "Hello there, how are you doing",
        "I'm fine tell me about",
    ]
    # TRP word probabilities
    t = model.tokenizer(text_list, include_end_ts=False)  # , return_tensors="pt")
    out = model(
        torch.tensor(t["input_ids"], device=model.device),
        torch.tensor(t["speaker_ids"], device=model.device),
    )
    out["probs"] = out["logits"].softmax(dim=-1)
    out["trp_probs"] = model.get_trp(out["probs"])
    p = model.tokenizer.extract_word_probs(t["input_ids"], out["trp_probs"])
    for k, v in zip(p["words"], p["probs"]):
        print(f"{k}: {v*100:.0f}%")

    # Generate future + TRP

    # Does seem to insert <ts> token? include_end_ts=True
    gen = generate(
        model,
        context=text_list,
        n_steps=5,
        top_p=0.9,
        top_k=-1,
        n_trajectories=20,
        strategy="sample",
        stop_at_eos=True,
    )
    print(gen.keys())

    n = 0
    for fut in gen["tokens"]:
        if "<ts>" in fut:
            n += 1
    print(f"Found {n} <ts> in {len(gen['tokens'])} samples")


if __name__ == "__main__":
    from os.path import join

    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Fresh Training
    fresh = False
    if fresh:
        model = TurnGPT(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            trp_projection_steps=args.trp_projection_steps,
            trp_projection_type=args.trp_projection_type,
            weight_loss=args.weight_loss,
            weight_eos_token=args.weight_eos_token,
            weight_regular_token=args.weight_regular_token,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            pretrained=args.pretrained,
            no_train_first_n=args.no_train_first_n,
            omit_dialog_states=args.omit_dialog_states,
        )
        model.init_tokenizer()
        model.initialize_special_embeddings()
    else:
        # projection
        chpt = join(
            "assets/TurnGPT_proj/version_0/checkpoints/epoch=45-val_loss=-3.37196.ckpt"
        )
        # only LM
        chpt = join(
            "assets/TurnGPT/version_0/checkpoints/epoch=11-val_loss=1.23640.ckpt"
        )
        model = TurnGPT.load_from_checkpoint(chpt).to("cuda")

    turn_list = [
        "Hello there I basically had the worst day of my life",
        "Oh no, what happened?",
        "Do you want the long or the short story?",
    ]

    gen = generate(
        model,
        context=turn_list,
        n_steps=200,
        top_p=0.9,
        top_k=-1,
        n_trajectories=20,
        strategy="sample",
        stop_at_eos=True,
    )
    # remove duplicates
    l = (gen["input_ids"][0] != -1).sum()
    G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
    for i, g in enumerate(gen["tokens"][1:]):
        if g not in G["tokens"]:
            l = (gen["input_ids"][i] != -1).sum()
            G["tokens"].append(g)
            G["probs"].append(gen["probs"][i][:l].cpu())

    #########################################################
    turn_list = [
        [
            "Hello there I basically had the worst day of my life",
            "Oh no, what happened?",
            "Do you want the long or the short story?",
        ],
        ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
    ]
    out = model.string_list_to_trp(turn_list)
    # for k, v in out.items():
    #     print(f"{k}: {v}")

    for b in range(out["trp_probs"].shape[0]):
        proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
        fig, ax = plot_trp(
            trp=out["trp_probs"][b].cpu(),
            proj=proj,
            text=out["tokens"][b],
            unk_token=model.tokenizer._tokenizer.unk_token,
            plot=True,
        )

from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip
import re

import torch
from torch.nn.functional import softmax

TurnGPT_TOKENS = {
    "pad_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
    ],
}

TurnGPT_TOKENS_TS = {
    "pad_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    "eos_token": "<ts>",
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
    ],
}


class SpokenDialogTokenizer:
    """
    Wrapper around Huggingface AutoTokenizer class.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument("--tokenizer", type=str, default="gpt2")

    @staticmethod
    def build_normalizer():
        normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex('[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
                Strip(),
            ]
        )
        return normalizer

    def __init__(self, pretrained_tokenizer="gpt2"):
        self.pretrained_tokenizer = pretrained_tokenizer
        self.normalizer = SpokenDialogTokenizer.build_normalizer()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        if pretrained_tokenizer == "gpt2":
            num_added_toks = self._tokenizer.add_special_tokens(TurnGPT_TOKENS)
            print(f"Extended {pretrained_tokenizer} tokenizer with {num_added_toks}")
            for special_token in self._tokenizer.additional_special_tokens:
                print("\t" + special_token)

            self.sp1_idx = self._tokenizer.convert_tokens_to_ids("<speaker1>")
            self.sp2_idx = self._tokenizer.convert_tokens_to_ids("<speaker2>")
            self.sp1_token = self._tokenizer.convert_ids_to_tokens(self.sp1_idx)
            self.sp2_token = self._tokenizer.convert_ids_to_tokens(self.sp2_idx)

    def __len__(self):
        return len(self._tokenizer)

    def encode(self, text, prefix=" ", **kwargs):
        """
        Encodes a text string into a list of input-indices. The prefix is used to (by default) insert a white-space at
        the beggining of the string in order to avoid the 'start of sentence' tokenization. When there is no white-space
        in from of a word it gets a different embedding than if there were. In spoken dialog with many turns the word
        that start the utterance should be encoded in the same way as if it was mentioned later.
        """
        text = self.normalizer.normalize_str(text)
        text = prefix + text
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs).strip()

    def convert_ids_batch_to_string(self, batch_inds):
        assert isinstance(batch_inds, torch.Tensor), "must provide tensor"
        output = []
        for x in batch_inds:
            t = self.convert_ids_to_tokens(x)
            output.append(self.convert_tokens_to_string(t))

        # TODO: hardcoded dangeruousness
        # create spaces in front of speaker shift characters
        for i, c in enumerate(output):
            output[i] = re.sub(r"(?!^)(\S)<speaker(\d)>", r"\g<1> <speaker\g<2>>", c)
        return output

    def format_special_chars(self, tokens):
        """https://github.com/jessevig/bertviz"""
        if self.pretrained_tokenizer == "gpt2":
            if isinstance(tokens, list):
                return [t.replace("Ġ", " ").replace("</w>", "").strip() for t in tokens]
            else:
                return tokens.replace("Ġ", " ").replace("</w>", "").strip()
        else:
            return tokens

    def ids_to_string(self, input_ids):
        return " ".join(
            self.format_special_chars(self.convert_ids_to_tokens(input_ids))
        )

    def turns_to_turngpt_tensors(
        self,
        turns,
        explicit_turn_shift=True,
        first_speaker=1,
        prefix=" ",
        add_end_speaker_token=True,
    ):
        assert isinstance(turns, list), "turns must be a list of strings"

        if first_speaker == 1:
            first_speaker_idx = self.sp1_idx
            second_speaker_idx = self.sp2_idx
        else:
            first_speaker_idx = self.sp2_idx
            second_speaker_idx = self.sp1_idx

        input_ids, speaker_ids = [], []
        for i, t in enumerate(turns):
            toks = self.encode(t, prefix=prefix)

            if i % 2 == 0:
                cur_speaker = first_speaker_idx
            else:
                cur_speaker = second_speaker_idx

            if explicit_turn_shift:
                input_ids.append(cur_speaker)
                speaker_ids.append(cur_speaker)

            input_ids += toks
            speaker_ids += [cur_speaker] * len(toks)

        if explicit_turn_shift and add_end_speaker_token:
            i += 1
            if i % 2 == 0:
                cur_speaker = first_speaker_idx
            else:
                cur_speaker = second_speaker_idx

            input_ids.append(cur_speaker)
            speaker_ids.append(cur_speaker)

        assert len(speaker_ids) == len(input_ids)
        return (
            torch.tensor(input_ids).unsqueeze(0),
            torch.tensor(speaker_ids).unsqueeze(0),
        )

    @torch.no_grad()
    def trp_from_logits(self, logits, sp1_idx, sp2_idx):
        probs = softmax(logits, dim=-1)
        trp, _ = probs[:, :, (sp1_idx, sp2_idx)].max(dim=-1)
        return trp


if __name__ == "__main__":

    tok = SpokenDialogTokenizer()
    text = "Hello there, how fun!"
    input_ids = tok.encode(text)

    print(input_ids)
    print(tok.decode(input_ids).strip())
    print(tok.convert_ids_to_tokens(input_ids))
    print(tok.convert_tokens_to_string(tok.convert_ids_to_tokens(input_ids)))

    tok.convert_ids_to_tokens(input_ids, format)

    turns = [
        'Yesterday Hello ther, "honey"',
        "godday... you are great",
        "Not as good as you!",
    ]

    input_ids, speaker_ids = tok.turns_to_turngpt_tensors(turns)
    print(input_ids)
    print(input_ids.shape)

    input_ids, speaker_ids = tok.turns_to_turngpt_tensors(turns, prefix="")
    print(input_ids)
    print(input_ids.shape)

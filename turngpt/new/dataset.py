from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers, normalizers, decoders, Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece

from ttd.tokenizer import (
    load_turngpt_tokenizer,
    get_special_tokens_dict,
)

from typing import List


class TurnTakingTokenizer(object):
    def __init__(
        self,
        pretrained="gpt2",
        special_token_dict={
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": [
                "<speaker1>",
                "<speaker2>",
            ],
        },
    ):
        # removes punctuation and capitalization
        self.normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex('[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
                Strip(),
            ]
        )
        self.tokenizer = load_turngpt_tokenizer(
            pretrained=pretrained,
            special_token_dict=special_token_dict,
        )
        self.sp1_idx = self.tokenizer.convert_tokens_to_ids("<speaker1>")
        self.sp2_idx = self.tokenizer.convert_tokens_to_ids("<speaker2>")
        self.pad_idx = self.tokenizer.pad_token_id

    def normalize(self, string):
        return self.normalizer.normalize_str(string)

    def tokenize_dialog(
        self, dialog: List[str], insert_speaker_tokens=True, prefix=" "
    ):
        input_ids = []
        speaker_ids = []
        for i, turn_string in enumerate(dialog):
            sp_token = self.sp1_idx if i % 2 == 0 else self.sp2_idx

            norm_string = self.normalize(turn_string)
            norm_string = prefix + norm_string
            ids = self.tokenizer.encode(norm_string)  # list
            sp_ids = [sp_token] * len(ids)

            if insert_speaker_tokens:
                ids = [sp_token] + ids
                sp_ids = [sp_token] + sp_ids
            input_ids += ids
            speaker_ids += sp_ids
        return torch.tensor(input_ids), torch.tensor(speaker_ids)

    def tokenize_string(self, string, prefix=" "):
        norm_string = self.normalize(string)
        norm_string = prefix + norm_string
        ids = self.tokenizer.encode(norm_string)  # list
        return torch.tensor(ids)


if __name__ == "__main__":

    tokenizer = TurnTakingTokenizer()
    string = 'Yesterday Hello there! Why so? They call me "solo"'
    dialog = ["Hi there! are you a student?", "yes i study math"]
    input_ids, speaker_ids = tokenizer.tokenize_dialog(dialog)
    print(input_ids.shape)
    print(speaker_ids.shape)

    input_ids = tokenizer.tokenize_string(string)
    tokenizer.tokenizer.convert_ids_to_tokens(input_ids)

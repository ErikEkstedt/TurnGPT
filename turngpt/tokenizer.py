from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Union
import torch
import re

import logging

logger = logging.getLogger(__name__)

TS_TOKENS = {
    "eos_token": "<ts>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}


class SpokenNormalizer:
    """
    Normalizer (as in the `tokenizers` framework) which removes punctuation, force lowercase, etc
    """

    def __init__(self):
        self.normalizer = SpokenNormalizer.build_normalizer()

    def normalize_string(self, s):
        s = self.add_whitespace_after_punctuation(s)
        return self.normalizer.normalize_str(s)

    def add_whitespace_after_punctuation(self, s):
        """
        Don't know how to do this with the `tokenizers` library.
        So simple regexp for now...

        Without this function:

            "hello,,,there;everybody.whats     how are you?"
            -> "hellothereeverybodywhats how are you" (once decoded)

        With:

            "hello,,,there;everybody.whats     how are you?"
            -> "hello there everybody whats how are you"

        """
        s = re.sub(r"[\,\.\:\;]+(\w+)", r" \1", s)
        return s

    @staticmethod
    def build_normalizer():
        normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),  # punctuation
                Replace(Regex(r"\s\s+"), " "),  # double spaces
                Strip(),
            ]
        )
        return normalizer


class SpokenDialogTokenizer(SpokenNormalizer):
    """
    A tokenizer wrapper for `AutoTokenizer.from_pretrained` which cleans/normalizes text
    strings, removes punctuations and creates `speaker_ids` (like TransferTransfo and similiar to Bert) where each utterance
    is imbued with a token corresponding to the correct speaker (<speaker1> and <speaker2>).

    Should work (kind of) like the normal `Tokenizers` in the `transformers` framework.

    IMPORTANT!!!
    ------------
    Do not have spaces prior to `eos_token`/<ts> in the complete dialog strings.
    The tokenizer inserts EMPTY SPACE!!!

    'hello there <ts>' -> ['hello', 'Ġthere' 'Ġ' '<ts>']

    this is bad!
    -----------------------------

    text_string = 'Yesterday Hello ther, "honey"<ts> godday... you are great<ts> Not as good as you!<ts>'
    o = tokenizer(text_string, return_tensors="pt")

    ----------------------------------------------------

    text_list = [
        'Yesterday Hello ther, "honey"',
        "godday... you are great",
        "Not as good as you!",
    ]
    o2 = tok(text_list, return_tensors="pt")
    print(o2["speaker_ids"] == o["speaker_ids"])
    for inps, spkrs in zip(o["input_ids"], o["speaker_ids"]):
        for i, s in zip(inps, spkrs):
            print(i.item(), s.item())

    ----------------------------------------------------

    list_of_lists = [text_list, text_list[:-1], text_list[:-2]]
    o = tok(text_string)
    o2 = tok(text_list)
    print(o2["speaker_ids"] == o["speaker_ids"])
    for i, s in zip(o["input_ids"], o["speaker_ids"]):
        print(i, s)


    """

    MODELS = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "gpt2",
    ]

    @property
    def unk_token(self):
        return self._tokenizer.unk_token

    @property
    def unk_token_id(self):
        return self._tokenizer.unk_token_id

    @property
    def eos_token(self):
        return self._tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    def __init__(self, pretrained_model_name_or_path="microsoft/DialoGPT-medium"):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path not in self.MODELS:
            print(
                f"WARNING: not tested for {pretrained_model_name_or_path} tread carefully!\n{self.MODELS}"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, max_model_input_sizes=None
        )

        # Set to large number to avoid warnings
        # Manually keep track of your models maximum input length
        self._tokenizer.model_max_length = 1e30

        # This goes in logging
        num_added_toks = self._tokenizer.add_special_tokens(TS_TOKENS)

        s = "Tokenizer initialization:\n"
        s += f"\tWe added {num_added_toks} tokens -> Special token map\n"
        for k, v in self._tokenizer.special_tokens_map.items():
            s += f"\t{k}: {v}\n"
        logger.info(s)

        # Turn-shift Token (eos_token)
        # self.eos_token = self._tokenizer.eos_token
        # self.eos_token_id = self._tokenizer.eos_token_id
        # self.unk_token = self._tokenizer.unk_token
        # self.unk_token_id = self._tokenizer.unk_token_id

        # Speaker Tokens
        self.sp1_token = TS_TOKENS["additional_special_tokens"][0]
        self.sp2_token = TS_TOKENS["additional_special_tokens"][1]
        self.sp1_token_id = self._tokenizer.convert_tokens_to_ids(self.sp1_token)
        self.sp2_token_id = self._tokenizer.convert_tokens_to_ids(self.sp2_token)

    def __repr__(self):
        return self._tokenizer.__repr__()

    def __len__(self):
        return len(self._tokenizer)

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        return_token_type_ids: bool = True,
        include_pre_space: bool = False,
        include_end_ts: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        SpokenDialogTokenizer tokenization.

        `text` can be either a String, a List of Strings, or a List of Lists of Strings. The behaviour of
        this function depends on the `single_dialog` flag.

        `text` is String:           representation of entire dialog (including eos_token)
        `text` is List[str]:        representation of turns in a dialog (no eos_tokens)
        `text` is List[List[str]]:  multiple dialogs (lists of strings) (no eos_tokens)

        """

        # List of lists
        if isinstance(text, list) and isinstance(text[0], list):
            ret = {}
            for text_list in text:
                o = self(
                    text_list,
                    include_pre_space=include_pre_space,
                    include_end_ts=include_end_ts,
                )

                for k, v in o.items():
                    if not k in ret:
                        ret[k] = []
                    ret[k].append(v)
            return ret
        # List of strings, a dialog: ['hello', 'hello to you']
        elif isinstance(text, List):
            dialog_string = ""
            if include_pre_space:
                dialog_string = " "
            dialog_string += self.normalize_string(text[0])
            if len(text) > 1:
                dialog_string += self.eos_token
                for text_string in text[1:-1]:
                    dialog_string += (
                        " " + self.normalize_string(text_string) + self.eos_token
                    )
                dialog_string += " " + self.normalize_string(text[-1])
            if include_end_ts:
                dialog_string += self.eos_token
            text = dialog_string
        else:
            text = self.normalize_string(text)

        encoding = self._tokenizer(
            text=text,
            **kwargs,
        )

        if return_token_type_ids:
            encoding["speaker_ids"] = self._extract_speaker_states(
                encoding["input_ids"]
            )
        return encoding

    def _extract_speaker_states(self, input_ids):
        # extract speaker states
        back_to_list = False
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).unsqueeze(0)  # with batch dim
            back_to_list = True
        # initialize with speaker 1
        speaker_ids = torch.ones_like(input_ids) * self.sp1_token_id
        batch, eos_idx = torch.where(input_ids == self.eos_token_id)
        for b in batch.unique():
            tmp_eos = eos_idx[batch == b]
            if len(tmp_eos) == 1:
                speaker_ids[b, eos_idx + 1 :] = self.sp2_token_id
            else:
                start = tmp_eos[0]
                for i, eos in enumerate(tmp_eos[1:]):
                    if i % 2 == 0:
                        sp = self.sp2_token_id
                        speaker_ids[b, start + 1 : eos + 1] = sp
                    start = eos
                if i % 2 == 1:  # add sp2 tokens after last eos if i is odd
                    speaker_ids[b, start + 1 :] = self.sp2_token_id

        if back_to_list:
            speaker_ids = speaker_ids.squeeze().tolist()
            if isinstance(speaker_ids, int):
                speaker_ids = [speaker_ids]

        return speaker_ids

    def idx_to_tokens(self, ids):
        def list_ids_to_string(ids):
            return [
                self.convert_tokens_to_string(t)
                for t in self.convert_ids_to_tokens(ids)
            ]

        # tokenize keep tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, list):
            if isinstance(ids[0], list):
                ret = [list_ids_to_string(ids_list) for ids_list in ids]
            else:
                ret = list_ids_to_string(ids)
        else:
            ret = self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))
        return ret

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_string(*args, **kwargs).strip()


if __name__ == "__main__":

    pretrained_model_name_or_path = "gpt2"
    tokenizer = SpokenDialogTokenizer(pretrained_model_name_or_path)

    turn_list = ["hello there how are you today?"]
    turn_list = ["hello", "good"]
    # turn_list = ["hello there how are you today?", "good", "great"]
    # turn_list = ["hello there how are you today?", "good", "great", 'yes']
    # turn_list = ["hello there how are you today?", "good", "great", 'yes', 'hello']
    # turn_list = ["hello there how are you today?", "good", "great", 'yes', 'hello', 'there']
    out = tokenizer([["hello", "bye"], ["hello", "bye", "you"]], include_end_ts=False)
    print(out)

    # double spaces
    s = "hello,,,there;everybody.whats<ts>     how are you?<ts>"
    print(s)
    t = tokenizer(s)
    print(tokenizer.decode(t["input_ids"]))

    s = "Hello there, how are you today?<ts> I'm doing good thank you!<ts> That's great<ts>"
    outputs = tokenizer(s)

    print(tokenizer.decode(outputs["input_ids"]))
    print(outputs["speaker_ids"])

    outputs["speaker"]

    turn_list = [
        "hello there how are you doing today?",
        "I'm doing very well thank you, how about you?",
        "well, I'm sad",
    ]

    i = tokenizer(turn_list, include_end_ts=False, include_pre_space=True)["input_ids"]
    d = tokenizer.decode(i)

    very_long_string = ""
    for i in range(150):
        very_long_string += "I'm doing very well thank you, how about you?"
    print(len(very_long_string.split(" ")))

    _ = tokenizer(very_long_string)

    turn_list = [
        "hello there how are you doing today?",
        "I'm doing very well thank you, how about you?",
        "well, I'm sad",
    ]
    tok_out = tokenizer(turn_list, include_end_ts=False)
    ids_list = tok_out["input_ids"]
    ids_list = tok_out["input_ids"]
    ids_tens = torch.tensor(tok_out["input_ids"])
    t1 = tokenizer.idx_to_tokens(ids_list)
    t2 = tokenizer.idx_to_tokens(ids_tens)
    t3 = tokenizer.idx_to_tokens(ids_list[0])

    outputs = tokenizer(list_of_lists, include_end_ts=False)

    output_strings = []
    for out in outputs["input_ids"]:
        output_strings.append(tokenizer.decode(out))

    assert output_strings == output_list_of_lists

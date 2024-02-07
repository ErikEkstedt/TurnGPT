import logging
import re
from typing import List, Tuple, Union

import torch
from tokenizers import Regex
from tokenizers.normalizers import (
    NFD,
    Lowercase,
    Replace,
    Sequence,
    Strip,
    StripAccents,
)
from transformers import AutoTokenizer

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

    def normalize_string(self, s: str) -> str:
        s = self.add_whitespace_after_punctuation(s)
        return self.normalizer.normalize_str(s)

    def add_whitespace_after_punctuation(self, s: str) -> str:
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


class SpokenDialogTokenizerBase(SpokenNormalizer):
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
        "gpt2",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
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

    def __init__(
        self,
        pretrained_model_name_or_path: str = "gpt2",
        normalization=True,
    ):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        if pretrained_model_name_or_path not in self.MODELS:
            print(
                f"WARNING: not tested for {pretrained_model_name_or_path} tread carefully!\n{self.MODELS}"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, max_model_input_sizes=None
        )
        self.normalization = normalization

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

        # Speaker Tokens
        self.sp1_token = TS_TOKENS["additional_special_tokens"][0]
        self.sp2_token = TS_TOKENS["additional_special_tokens"][1]
        self.sp1_token_id = self._tokenizer.convert_tokens_to_ids(self.sp1_token)
        self.sp2_token_id = self._tokenizer.convert_tokens_to_ids(self.sp2_token)

        self.new_word_char = "Ġ"

    def __repr__(self) -> str:
        return self._tokenizer.__repr__()

    def __len__(self) -> int:
        return len(self._tokenizer)

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

    def normalize(self, string: str) -> str:
        if self.normalization:
            return self.normalize_string(string)
        return string


class SpokenDialogTokenizer(SpokenDialogTokenizerBase):
    def _is_list_of_lists(self, text) -> bool:
        return isinstance(text, list) and isinstance(text[0], list)

    def _is_list_of_strings(self, data: Union[str, List[str], List[List[str]]]) -> bool:
        return isinstance(data, list) and all(isinstance(item, str) for item in data)

    def format_string(
        self, text: str, include_end_ts: bool, include_pre_space: bool
    ) -> str:
        text = self.normalize(text)
        if include_pre_space and not text.startswith(" "):
            text = " " + text
        if include_end_ts and not text.endswith(self.eos_token):
            text += self.eos_token
        return text

    def format_list_of_strings(
        self, list_of_strings: List[str], include_end_ts: bool, include_pre_space: bool
    ) -> str:
        text_string = f"{self.eos_token} ".join(list_of_strings)
        return self.format_string(
            text_string,
            include_end_ts=include_end_ts,
            include_pre_space=include_pre_space,
        )

    def _extract_speaker_states(self, input_ids):
        """
        extract speaker states
        """
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
                i = 0
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
        """"""

        def list_ids_to_string(ids):
            return self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))

        # tokenize keep tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, int):
            ids = [ids]

        if isinstance(ids, list):
            if isinstance(ids[0], list):
                ret = [list_ids_to_string(ids_list) for ids_list in ids]
            else:
                ret = self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))
        else:
            ret = self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))
        return ret

    def extract_word_probs(self, input_ids, probs):
        """ """

        is_tensor = False
        if isinstance(input_ids, torch.Tensor):
            assert input_ids.ndim == 1, f"input_ids must be 1D. Got {input_ids.ndim}"

        if isinstance(input_ids, list):
            assert isinstance(input_ids[0], int), "input_ids must be 1D list of ints"

        if isinstance(probs, torch.Tensor):
            assert probs.ndim == 1, f"probs must be 1D. Got {probs.ndim}"
            is_tensor = True

        if isinstance(probs, list):
            assert isinstance(
                probs[0], (float, int)
            ), "input_ids must be 1D list of floats"

        tok_list = self.convert_ids_to_tokens(input_ids)
        words = [tok_list[0]]
        p_words = [probs[0]]
        for i in range(1, len(tok_list)):
            if (
                tok_list[i].startswith(self.new_word_char)
                or tok_list[i] == self.eos_token
            ):
                p_words.append(probs[i])
                words.append(tok_list[i].replace(self.new_word_char, ""))
            else:
                p_words[-1] = max(p_words[-1], probs[i])
                words[-1] += tok_list[i].replace(self.new_word_char, "")

        if is_tensor:
            p_words = torch.tensor(p_words)
        return {"words": words, "probs": p_words}

    def word_probs_filter_last_utterance(
        self, w: List[str], p: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor]:
        last_ts_idx = w[::-1].index("<ts>") if self.eos_token in w else 0
        return w[-last_ts_idx:], p[-last_ts_idx:]

    def get_prefixes(self, tokens: List[str]) -> Tuple[List[int], List[str]]:
        prefixes = []
        n_words_left = []
        for future in tokens:
            n = future.split(self.eos_token)
            user_continuation = n[0].strip()
            n_words_left.append(len(user_continuation.split()))
            if len(n) > 1 and n[1] != "":
                prefix = n[1].strip()
                prefixes.append(prefix)
                # print(f"{i}. {user_continuation} --------- {prefix}")
        return n_words_left, prefixes

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        return_token_type_ids: bool = True,
        include_pre_space: bool = False,
        include_end_ts: bool = True,
        **kwargs,
    ):
        """
        SpokenDialogTokenizer tokenization.

        `text` can be either a String, a List of Strings, or a List of Lists of Strings. The behaviour of
        this function depends on the `single_dialog` flag.

        `text` is String:           representation of entire dialog (including eos_token)
        `text` is List[str]:        representation of turns in a dialog (no eos_tokens)
        `text` is List[List[str]]:  multiple dialogs (lists of strings) (no eos_tokens)
        """
        if self._is_list_of_lists(text):
            # Checks for a list of lists
            ret = {}
            for list_of_strings in text:
                o = self(
                    list_of_strings,
                    include_pre_space=include_pre_space,
                    include_end_ts=include_end_ts,
                    **kwargs,
                )
                for k, v in o.items():
                    if k not in ret:
                        ret[k] = []
                    ret[k].append(v)
            return ret

        if self._is_list_of_strings(text):
            text = self.format_list_of_strings(
                text, include_end_ts=include_end_ts, include_pre_space=include_pre_space
            )
        elif isinstance(text, str):
            text = self.format_string(
                text, include_end_ts=include_end_ts, include_pre_space=include_pre_space
            )
        else:
            raise ValueError(
                f"Invalid input type: {type(text)}. Expects: str, List[str], List[List[str]]"
            )

        # Encoder the string
        encoding = self._tokenizer(text=text, **kwargs)

        # Extract speaker_ids
        if return_token_type_ids:
            encoding["speaker_ids"] = self._extract_speaker_states(
                encoding["input_ids"]
            )
        return encoding


def original_tokenizer():
    T1 = AutoTokenizer.from_pretrained("gpt2", max_model_input_sizes=None)
    T2 = SpokenDialogTokenizer("gpt2")
    text = "Hello how are u"
    inp1 = T1(text)["input_ids"]
    inp2 = T2(text)["input_ids"]
    print("words: ", len(text.split()))
    print("text: ", text)
    print("inp1: ", len(inp1), inp1)
    print("inp2: ", len(inp2), inp2)
    t1 = T1.convert_ids_to_tokens(inp1)
    t2 = T2.convert_ids_to_tokens(inp2)
    i1 = T1.convert_tokens_to_ids(t1)
    i2 = T2.convert_tokens_to_ids(t2)
    it1 = T1.convert_tokens_to_string(t1)
    it2 = T2.convert_tokens_to_string(t2)
    print("t1: ", t1)
    print("t2: ", t2)
    print("i1: ", i1)
    print("i2: ", i2)
    print("it1: ", it1)
    print("it2: ", it2)


def test():

    text_string = "hello there how are you today?"
    ans_string_inp = [31373, 612, 703, 389, 345, 1909]
    tokenizer = SpokenDialogTokenizer("gpt2")
    t = tokenizer([text_string, text_string], return_tensors="pt")
    text_string = "Yesterday i had tommorows intervention"
    # text_string = "hello there how are you today?"
    t = tokenizer(
        ["Yesterday i had tommorows intervention", "Oh is that so but yesterday?"],
        return_tensors="pt",
    )
    input_ids = t["input_ids"][0]
    probs = torch.arange(t["input_ids"].shape[-1])
    print("input_ids: ", input_ids)
    print("probs: ", probs)
    wp = tokenizer.extract_word_probs(t["input_ids"][0], probs)
    print(wp["words"])
    print(wp["probs"])

    tok = tokenizer.idx_to_tokens(612)
    tok = tokenizer.idx_to_tokens([612])
    tok = tokenizer.idx_to_tokens([612, 703])

    # encode a string
    t = tokenizer(
        text_string,
        return_token_type_ids=False,
        include_pre_space=False,
        include_end_ts=True,
    )
    print(t["input_ids"])

    print(t["input_ids"] == ans_string_inp)

    # Encode a list of strings
    t = tokenizer([text_string])
    print(t["input_ids"] == ans_string_inp)

    text_list = [
        "Yesterday i had tommorows intervention",
        "Oh is that so but yesterday?",
        "I don't know",
    ]

    tokenizer = SpokenDialogTokenizer("gpt2")

    # Use case: use the model in a SDS to infer upcoming TRPs
    # Assume that we don't add <ts> tokens to the end of the utterances

    t = tokenizer(text_list, include_end_ts=False, return_tensors="pt")  # (1, N)
    probs = torch.arange(t["input_ids"].shape[-1]).unsqueeze(0)

    wp = tokenizer.extract_word_probs(t["input_ids"].squeeze(0), probs.squeeze(0))
    print(wp["words"])
    print(wp["probs"])

    ww, pp = tokenizer.word_probs_filter_last_utterance(wp["words"], wp["probs"])

    print(ww)
    print(pp)

    context = ["Is your name Alice?", "Yes, I am she", "Okay"]
    t = tokenizer(text_list, include_end_ts=False, return_tensors="pt")  # (1, N)

    tokens = [
        " what's going on<ts> everything",
        " what's wrong<ts> there's",
        " what's going on<ts> oh",
        " what you're doing here<ts>",
        "<ts> i don't know what",
        " what's going on<ts> nothing",
        " what's going on<ts> i",
        " what's wrong<ts> i don",
        " what's wrong<ts> i don",
        " what's going on<ts> i",
        " what's going on<ts> i",
        " what's going on you look",
        " what's going on<ts> my",
        " what you're doing here<ts>",
        " about yourself<ts> well i'm",
        "<ts> i was just wondering why",
        " what's going on<ts> there",
        "<ts> oh just a bit of",
        " what's going on<ts> i",
        " what's wrong<ts> oh nothing",
    ]

    t = tokenizer(tokens, include_end_ts=False, return_tensors="pt")["input_ids"]

    tokenizer = SpokenDialogTokenizer("gpt2")

    # Get prefix LM stuff
    # Assume that it is the ongoing speech from the user
    # Find first <ts> and the following agent utterance
    nwords, prefix = tokenizer.get_prefixes(tokens)


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

    turn_list = [
        "hello there how are you doing today?",
        "I'm doing very well thank you, how about you?",
        "well, I'm sad",
    ]

    i = tokenizer(turn_list, include_end_ts=False, include_pre_space=False)["input_ids"]
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

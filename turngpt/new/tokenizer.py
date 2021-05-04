from tokenizers import Tokenizer
from tokenizers import pre_tokenizers, normalizers, decoders, Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece

from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from os.path import join

try:
    from tokenizers import Tokenizer
except ImportError:
    raise ImportError(
        "TurnTaking requires Huggingface tokenizers installed. \n"
        "pip install tokenizers"
    )


SPECIAL_TOKENS = [
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "[S0]",
    "[S1]",
]


def build_spoken_dialog_tokenizer():
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace()])
    tokenizer.normalizer = normalizers.Sequence(
        [
            NFD(),
            Lowercase(),
            StripAccents(),
            Replace(Regex('[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
            Strip(),
        ]
    )
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer


class PosExtractionNLTK(object):
    POS_LIST = ["WP", "NN", "NNP", "VBP", "VBP"]

    def __init__(self, pos_list=None):
        self.pos_list = pos_list
        if pos_list is None:
            self.pos_list = PosExtractionNLTK.POS_LIST

    def __call__(self, string):
        pos = pos_tag(word_tokenize(string))
        valued_pos = []
        for pos_name in self.pos_list:
            valued_pos += [p[0] for p in pos if p[1] == pos_name]
        return valued_pos


if __name__ == "__main__":
    from glob import glob
    from ttd.utils import read_json

    import torch
    from torch.nn.utils.rnn import pad_sequence

    dtokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    btokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenizer = build_spoken_dialog_tokenizer()
    vocab_files = glob(join("text_files_for_bpe", "*.txt"))
    vocab_size = 8000
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train(trainer, vocab_files)

    POS = PosExtractionNLTK()

    s0 = 'Hello! I am "cool"... You are?'
    s1 = "Hi! I am just (well) me..."

    s0 = tokenizer.normalizer.normalize_str(s0)
    s1 = tokenizer.normalizer.normalize_str(s1)

    bert_toks = btokenizer(s0, s1, return_tensors="pt")

    print(bert_toks)
    print(bert_toks.shape)

    string = s1 + " " + s0
    p = POS(string)
    p_ids = btokenizer.encode(p, add_special_tokens=False)
    print(p)
    print(p_ids)

    # READ DIALOG
    root = "data/taskmaster/dialogs_turn_level"
    files = glob(join(root, "*.json"))
    dialog = read_json(files[0])

    current = [], []
    context = [tokenizer.normalizer.normalize_str(dialog[0]["text"])]

    for d1 in dialog[1:]:
        s1 = tokenizer.normalizer.normalize_str(d1["text"])
        current.append(s1)

    context = [tokenizer.normalizer.normalize_str(d["text"]) for d in dialog[:-1]]
    current = [tokenizer.normalizer.normalize_str(d["text"]) for d in dialog[1:]]

    batch = btokenizer(
        context, current, padding=True, truncation=True, return_tensors="pt"
    )

    for s0, s1 in zip(context, current):
        string = s0 + " " + s0
        p = POS(string)
        print(string, p)

    btokenizer.decode(batch["input_ids"])

    info_labels = torch.zeros_like(b)
    for i, b in enumerate(bert_toks):
        if b in p_ids:
            info_labels[i] = 1

    # Special start token here...
    pairs_input_ids = []
    pairs_type_ids = []
    pairs_pos_ids = []
    s0 = ""
    for turn_ind in range(len(dialog)):
        s1 = dialog[turn_ind]["text"]
        # print(turn_ind, s0, s1)
        tmp_state = tokenizer.encode(s0, s1)
        pairs_input_ids.append(torch.tensor(tmp_state.ids))
        pairs_type_ids.append(torch.tensor(tmp_state.type_ids))
        # pos labels
        string = s0 + " " + s1
        string = tokenizer.normalizer.normalize_str(string)
        print(POS(string))
        s0 = s1

    pairs_input_ids = pad_sequence(
        pairs_input_ids, batch_first=True, padding_value=-100
    )
    pairs_type_ids = pad_sequence(pairs_type_ids, batch_first=True, padding_value=-100)

    pairs_pos_ids = pad_sequence(pairs_pos_ids, batch_first=True, padding_value=-100)

    print("ds_input_ids: ", tuple(ds_input_ids.shape))
    print("ds_type_ids: ", tuple(ds_type_ids.shape))

    t = tokenizer.encode(s0, s1)
    tokenizer.post_process(s0, s1)

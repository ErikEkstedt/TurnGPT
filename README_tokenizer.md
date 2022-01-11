# Tokenizer

1. Convenient
    - Coherent with [examples](https://huggingface.co/transformers/preprocessing.html)
    - GPT2 and DialoGPT tokenizers.
2. Clean 
    - lower case
    - remove punctuation
3. Provide function to extract `speaker_ids` from dialog data.
4. Usable for [datasets](https://huggingface.co/docs/datasets) map function.
5. Improve whenever possible to be faster and better at removing punctation...


## Normalizers

Using the [tokenizers](https://huggingface.co/docs/tokenizers) library and specifically it's `Normalalizer`. 
Appliying the normalizer before tokenization by `AutoTokenizer`.


```python
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)
normalizer = Sequence(
    [
        NFD(),
        Lowercase(),
        StripAccents(),
        Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
        Strip(),
    ]
)
```

## Embeddings

```python
TS_TOKENS = {
    "eos_token": "<ts>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}
```

## Use

```python
from convlm.tokenizer import SpokenDialogTokenizer

pretrained_model_name_or_path="microsoft/DialoGPT-small"
tokenizer = SpokenDialogTokenizer(pretrained_model_name_or_path)

# tokenizer.eos_token: '<ts>'
# tokenizer.eos_token_id: 50257

# tokenizer.sp1_token: '<speaker1>'
# tokenizer.sp1_token_id: 50258

# tokenizer.sp2_token: '<speaker2>'
# tokenizer.sp2_token_id: 50259

text_list = [
    'Yesterday Hello ther, "honey"',
    "godday... you are great",
    "Not as good as you!",
]

outputs = tokenizer(text_list)

# print(outputs.keys())
# >>> dict_keys(['input_ids', 'attention_mask', 'speaker_ids'])

# input_ids: word embedding indices
# >>> input_ids: [8505, ...,  220, 50257, 5770, ..., 50257]

# attention_mask: mask to omit `pad_token` in loss
# >>> attention_mask: [1, ...,  1, 1, 1, ..., 1]

# speaker_ids: dialog state embeddings corresponind to speaker id (binary)
# >>> speaker_ids: [50258, ..., 50259, ..., 50258]

decoded_input = tokenizer.decode(outputs['input_ids']) # arugment must be a list
# print(decoded_input)
# >>> 'yesterday hello ther honey <ts> godday you are great <ts> not as good as you <ts>'

```

## Tests

TBD


-----------

#### Detective work in transformers lib

* `PreTrainedTokenizerBase`:
  * https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils_base.py#L1432
* `PreTrainedTokenizerFast`:
  * https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils_fast.py#L76
* `PreTrainedTokenizerFast` depends on `PreTrainedTokenizerBase` 

tokenizer.__call__() checks some conditions and then calls:

* `self.batch_encode_plus`
* `self.encode_plus`

The FASTTokenizers then have their own `self._batch_encode_plus` and `self._encode_plus` which are called in the 
corresponding functions without the "_"-prefix.

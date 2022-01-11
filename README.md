# TurnGPT

TurnGPT is a finetuning setup for training (conversational turn-taking)
Language Models using pretrained model weights from strong Pretrained Language
Models, varyants of
[GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
decoder only Transformer Language Model.

TurnGPT is a turn-taking focused language model implementation of GPT
Transformers.  The original model utilized speaker states as in
[TransferTransfo](https://arxiv.org/pdf/1901.08149.pdf), where each speaker
utterances are encoded with a specific speaker token, similar to position
tokens and BERTs sentence tokens. I do not think that is hugely important for
performace (but I must test).

TurnGPT as aimed for use in Spoken Dialog Systems, for turn-taking, and so we
focus on text as is commonly acquired when implementing an SDS. For this
project, this means that all text should be lower-case and certain punctations
should be filtered out, text as commonly returned from ASR services (or local
models). The custom tokenizer normalizes input text strings to that format
automatically. It also construct speaker-representations as well as inputs a
turn-shift token between turns.


# Installation

* Create conda env: `conda create -n turngpt python=3`
  - source env: `conda source turngpt`
* PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
* Dependencies: `pip install -r requirements.txt`
* Install [Datasets turn-taking](https://github.com/ErikEkstedt/datasets_turntaking)
    - clone repo, cd to repo, and install dependencies: `pip install -r requirements.txt`
    - install repo: `pip install -e .`
* cd into this repo and install turngpt: `pip install -e .`


# Notes 

* Warning: This "simplified" branch does not incoorperate spoken-dialogs (switchboard, maptask) because the data have to be downloaded separetely.
* Warning: The Analysis from the paper is not provided in this repo but simpler training of the model.
* Addon: Added projection head to model a turn-shift in the next `N` tokens.
  - This effects training (how to combine/scale losses to work well together?)


# Code


## Tokenizer

The tokenizer is called SpokenDialogTokenizer see the [README](README_tokenizer.md)
1. Wrapper around [GPT2Tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer) and [DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html) (which I believe are almost the same)
2. Normalizes raw text (reg-exp, remove punctuation, lower-case), using huggingface [tokenizers](https://huggingface.co/docs/tokenizers).
    - raw text are lists of utterances ([String])
3. Automatically assigns turn-shift token `<ts>` and (optional) speaker-tokens `<speaker1>, <speaker2>` appropriately
4. (Optional) initialized as average of ["?", "!", ".", ","] embeddings.


see `README_tokenizer.md` for further information.

#### Simple use

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

## Model

The [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/) model is a wrapper which loads pretrained models (GPT2, DialoGPT) from [huggingface transformers library](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Starting Fresh 

An un-trained TurnGPT model, loads pre-trained weights by default, and includes the tokenizer.

```python
  from argparse import ArgumentParser
  from convlm.turngpt import TurnGPT

  parser = ArgumentParser()
  parser = TurnGPT.add_model_specific_args(parser)
  args = parser.parse_args()

  # print out args
  for k, v in vars(args).items():
      print(f"{k}: {v}")

  # Fresh Initialization
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

  # These must be called on a freash initialization (later done when loading the model)
  # on checkpoint-save the `tokenizer` is saved with the model.
  # on checkpoint-load the `tokenizer` is loaded and the weights extended automatically
  model.init_tokenizer()  # required for fresh model (saved on checkpoint)
  model.initialize_special_embeddings()  # required for fresh model (also performed on load_checkpoint)
  model.print_parameters()

  print(model.tokenizer)
  # PreTrainedTokenizerFast(name_or_path='gpt2', vocab_size=50257,
  # model_max_len=1024, is_fast=True, padding_side='right',
  # special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<ts>',
  # 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>',
  # 'additional_special_tokens': ['<speaker1>', '<speaker2>']})

```

### Load model

```python
# Load trained model e.g.
model = TurnGPT.load_from_checkpoint("PATH/TO/checkpoint.ckpt")
```

## TRP or Turn-shift

Use the model to extract TRP given `strings`, `list of strings` and `list of list of strings`.

```python
  # Example use
  turn_list = [
      "Hello there I basically had the worst day of my life",
      "Oh no, what happened?",
      "Do you want the long or the short story?",
  ]
  turn_list2 = [
      "Hello there I basically had the worst day of my life",
      "Oh no, what happened?",
  ]
  multiple = [turn_list, turn_list2]

  # Get trp from a text string
  out = model.string_list_to_trp(turn_list[0], add_post_eos_token=True)

  # Get trp from a text list
  out = model.string_list_to_trp(turn_list)

  # Get trp from a list of text lists
  out = model.string_list_to_trp(multiple)

  # out: dict_keys(['logits', 'past_key_values', 'probs', 'trp_probs', 'tokens'])

  # Simple Plot
  import matplotlib.pyplot as plt

  def plot_trp(P, text):
      fig, ax = plt.subplots(1, 1)
      x = torch.arange(len(P))
      ax.bar(x, P)
      ax.set_xticks(x)
      ax.set_xticklabels(text, rotation=60)
      ax.set_ylim([0, 1])
      plt.pause(0.01)
      return fig, ax

  fig, ax = plot_trp(out["trp_probs"][0], out["tokens"][0])
```

### Code Sources
* [transformers](https://huggingface.co/transformers)
* [tokenizers](https://huggingface.co/docs/tokenizers)
* [datasets](https://huggingface.co/docs/datasets)
* [pytorch-lightning](https://pytorch-lightning.readthedocs.io/)


### Knowledge sources
* [TransferTransfo: A Transfer Learning Approach for Neural Network Based Conversational Agents](https://arxiv.org/pdf/1901.08149.pdf)
  * Huggingface [blog](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
  * Task-oriented dialogs [Hello, It's GPT-2 - How Can I Help You?](https://arxiv.org/pdf/1907.05774.pdf)
* [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT2](), [GPT3]()
* [DialoGPT](https://aclanthology.org/2020.acl-demos.30.pdf)

# Reference

```latex
@inproceedings{ekstedt-skantze-2020-turngpt,
    title = "{T}urn{GPT}: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog",
    author = "Ekstedt, Erik  and
      Skantze, Gabriel",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.268",
    doi = "10.18653/v1/2020.findings-emnlp.268",
    pages = "2981--2990",
    abstract = "Syntactic and pragmatic completeness is known to be important for turn-taking prediction, but so far machine learning models of turn-taking have used such linguistic information in a limited way. In this paper, we introduce TurnGPT, a transformer-based language model for predicting turn-shifts in spoken dialog. The model has been trained and evaluated on a variety of written and spoken dialog datasets. We show that the model outperforms two baselines used in prior work. We also report on an ablation study, as well as attention and gradient analyses, which show that the model is able to utilize the dialog context and pragmatic completeness for turn-taking prediction. Finally, we explore the model{'}s potential in not only detecting, but also projecting, turn-completions.",
}
```

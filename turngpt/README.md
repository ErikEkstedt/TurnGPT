# TurnGPT

TurnGPT is a finetuning setup for training (conversational turn-taking)
Language Models using pretrained model weights from strong Pretrained Language
Models, varyants of
[GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
decoder only Transformer Language Model.

# Transformers

In language translation there exist some underlying structure of the problem,
translate the words (from vocabulary $V_A$) of one sentence in langugage $L_A$,
that is the underlying meaning (in concept space), to another language $L_B$
using words in that vocabulary $V_B$, that makes it suitable for an
Encoder-Decoder framework of thinking.

The framework of thinking means how to model the information flow of
transmitting a message, or what you learn in introduction to Information
Theory.

* Encoder/Decoder must process inputs of varying length and $N_A$
  - RNN became widely popular to use. Inductive bias for variable length time series problems.
  - len(sentence a) == len(sentence b) is not always the case. How common? I don't know... not very? seems improbable..
* In practice this means to represent the information of the translated message in some vector-space by the Encoder. 
  Which is then processed by the Decoder to output the translation
* Difficult to compress information
  - It is difficult to learn such a vector
  - To compress the representations of $N_A$ tokens, into a fixed sized conceptual representation, should be hard.
* It works better if the decoder can attend to each step of the input
  - We are not required to compress to such an extent

### Attention

- Attention is the code mechanism that operates on all (encoded) input representations, at each step of decoding
- based on similarity scores on **Key** and **Query** vectors of all tokens which
- determines how to modify the represented **Value** in the current decoder step.


In their, now infamous paper, [Attention is all you
Need](https://arxiv.org/pdf/1706.03762.pdf) people at Google used an Attention
mechanism, on MLP based models, turning *attention* away from the very common
Recurrent Neural Networds. Setting the record (State of the Art) on common
translation tasks. The proposed 

$$
Q = W_QX, \\
K = W_KX, \\
V = W_VX, \\
Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_k}}) V \\
$$

This approach don't "throw away" information like RNN (always keep entire
history in current state) and the conceptual translation does not get lost but
are more distributed (not as compressed) defined by similarity connections and
values of all past representations. That is very useful when generating
concepts in one vocabulary conditioned on a sentence from another vocabulary.

* Easier to translate because the meaning must not be as compressed as a single vector.

### GPT and Bert

What is the different between the Encoder and the Decoder of a Transformer?

Google modelled "contextualized word embeddings" (BERT), using only the Encoder
of the orginal transformer, which improved the performance on many downstream
tasks. The model itself included a natural way to finetung for specific tasks
and became very popular. OpenAI instead applied only the Decoder part of the
original transformer specifically for language modeling, predicting the next
token given the history, with SOTA comparable results. Similarily to Google
they also introduced ideas on how to finetune such LMs on downstream tasks.
They made popular the notion that many problems in NLP can be restructed in the
context of language modelling.


# TurnGPT

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
  )

  # These must be called on a freash initialization (later done when loading the model)
  # on checkpoint-save the `tokenizer` is saved with the model.
  # on checkpoint-load the `tokenizer` is loaded and the weights extended automatically
  model.init_tokenizer()  # required for fresh model (saved on checkpoint)
  model.initialize_special_embeddings()  # required for fresh model (also performed on load_checkpoint)

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

# Code Sources
* [transformers](https://huggingface.co/transformers)
* [tokenizers](https://huggingface.co/docs/tokenizers)
* [datasets](https://huggingface.co/docs/datasets)
* [pytorch-lightning](https://pytorch-lightning.readthedocs.io/)


# References
* [TransferTransfo: A Transfer Learning Approach for Neural Network Based Conversational Agents](https://arxiv.org/pdf/1901.08149.pdf)
  * Huggingface [blog](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
  * Task-oriented dialogs [Hello, It's GPT-2 - How Can I Help You?](https://arxiv.org/pdf/1907.05774.pdf)
* [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT2](), [GPT3]()
* [DialoGPT](https://aclanthology.org/2020.acl-demos.30.pdf)
* [Rasa Whiteboard](https://www.youtube.com/watch?v=wWNMST6t1TA&list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb)


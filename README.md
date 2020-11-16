# TurnGPT

TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog

Built using [pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning).


## Installation

* Install pytorch (1.6.0) and torchaudio (0.6.0)
* Install TTD
  * `git clone https://github.com/ErikEkstedt/TTD.git`
  * `cd $PATH_TO_TTD`
  * `pip install -r requirements.txt`
  * `pip install -e .`
* Clone this repo `git clone https://github.com/ErikEkstedt/TurnGPT.git`
  * `cd $PATH_TO_TurnGPT`
  * `pip install -r requirements.txt`
  * `pip install -e .`


## Train

```bash
python turngpt/main.py \
  --model pretrained \
  --datasets taskmaster metalwoz multiwoz coached persona dailydialog \
  --chunk_size 512 \
  --gpus 1 \
  --batch_size 2 \
```

## Eval

```bash
python turngpt/eval.py \
  --model pretrained \
  --checkpoint $CHECKPOINT_PATH
  --tokenizer $TOKENIZER_PATH
  --datasets persona \
  --chunk_size 512 \
  --batch_size 2 \
  --context_ablation \  (optional)
  --context_attention \ (optional)
  --context_ig \        (optional)
  --prediction_hist \   (optional)
```

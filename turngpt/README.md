# TurnGPT


The pytorch-Lightning Module is defined in `TurnGPT.py` and the training script in `main.py`


## Training

The pretraining and finetuning is done in the same way (with or without proximity model/acoustic model)

### Pre-training

The pre-training consists of training the Language Model, which by default, is the the GPT2
implementation by Huggingface. We train on written corpora to learn the distribution over
non-capitalized and non-punctuated text with the addition of the speaker/speaker-shift tokens.

Available datasets:
* Taskmaster 1 + 2
* MultiWoz
* MetaLWoz
* Coached
* Persona
* Daily Dialog
* Empathetic

To simply train the langugage model run:

```bash
python /path/to/main.py \
  --gpus1 \  # pytorch lightning gpus usage
  --datasets persona empathetic dailydialog coached
```

### Finetune

This step swiches domain into spoken dialog. In order to fully understand when turn-shifts are about
to occur we should also include acoustic information.


```bash
python /path/to/main.py \
  --gpus1 \
  --datasets switchboard \  
  --audio \  # add this flag to include audio information
  --checkpoint /Path/to/pre-trained-model \  # load the pretrained model weights
  --acoustic_model transformer \
  --proximity_model transformer \
  --prox_horizon 1 \  # -> list of horizons to predict
```



### Spoken Dialog Training

Given a pre-trained Language Model on some text-based dialog corpora we now include both audio
information and an extra training objective. 

#### Acoustic Information

The acoustic informaiont consists of the audio associated with the dialogs. The acoustic information
comes in two forms the waveform data or pre-extracted prosodic features (pitch, voiced,
zero-crossing-rate, rms). The model trains of acoustic segment (of word_audio_segment_time,
default=1s) for each token. Each segment ends when the word associated with the token ends. This
means that some subword tokens will get information about "the future", however, this data would
always be available to the model in any spoken interaction (we would only process a word once its
completed).

The pre-extracted prosodic features are normalized over each speaker channel and dialog.


#### Acoustic Model


The acoustic model consists of several parts. first an Conv1d encoder that

###### Feature Based Model

1. Conv (independent) encoder (n_feature representations for each token)
2. Feature attention (condense to a single representation for each word)
3. Uni-directional Transformer over audio-word-representations (operates over the sequence of audio token
   representation)

The feature based model consists of a feature-independent Conv1D model with 3 layers (by default)
and a fully connected layer projecting encoding each independent feature representation. These
representations are then fed through

#### Modal Mixer

The modal mixer is a single attention layer wich attend over the word or acoustic hidden states. In
a sense this should make the model choose which information to use at any given moment in time. It
also provides a way to infer where and when the model uses acoustic information etc.

```bash
Zw = LM(S)
Za = AcousticModel(X)
Zm = ModalMix(Za, Zw)
```


#### Proximity Model

The goal of the proximity model is to simply output a probability over how close a turn-shift is at
any token in a sequence. The proximity model may be trained with multiple horizons. A horizon of 1
means that the model predicts weather the next token is a turn-shift or not and similarly a horizon
of 3 means to predict weather a turn-shift occurs in the upcomming three tokens.

The proximity model can operate on only the last hidden states of the LM (GPT2) or in combination
with the hidden states from an acoustic model.

The default architecture for the proximity model is a uni-directional transformer model with a
single layer. In other words it works as adding an additional layer to the GPT2 structure trained to
output proximity probabilities.


```bash
Zm = MM(Zw, Za)
proximity_logits = ProximityModel(Zm)
```




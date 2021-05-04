from argparse import ArgumentParser

import torch
import torch.nn as nn
from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.utils import get_best_response, predict_trp
from turngpt.tokenizer import SpokenDialogTokenizer

import matplotlib.pyplot as plt


def incremental_turn_prediction(turns, model, tokenizer, N=100, n_tokens=3):
    all_w = [t.split() for t in turns]
    all_p = []
    for i, turn in enumerate(turns):
        if i % 2 == 1:  # user text
            context = turns[:i]
            words = turn.split()
            user_turn = ""
            user_pred = []
            for wi, word in enumerate(words):
                if wi == 0:
                    user_turn = word
                else:
                    user_turn += " " + word
                tmp_data = context + [user_turn]
                input_ids, speaker_ids = tokenizer.turns_to_turngpt_tensors(
                    tmp_data, add_end_speaker_token=False
                )
                p = predict_trp(model, input_ids, speaker_ids, N=N, n_tokens=n_tokens)
                user_pred.append(p)
            all_p.append(user_pred)
        else:
            all_p.append([0] * len(turn.split()))

    words, ps = [], []
    for wturn, pturn in zip(all_w, all_p):
        words += ["EOT"] + wturn
        ps += [0] + pturn
    return words, ps


def plot_predictions(words, ps, figsize=(9, 6), plot=True):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = torch.arange(len(words))
    ax.bar(x, ps)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=60)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(x)])
    plt.tight_layout()
    if plot:
        plt.pause(0.1)
    return fig, ax


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/erik/projects/checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981_V2.ckpt",
        # default="/home/erik/projects/checkpoints/pretrained/PDEC/checkpoints/epoch=3-val_loss=1.98388.ckpt",
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    tokenizer = SpokenDialogTokenizer(pretrained_tokenizer="gpt2")
    model = TurnGPTModel.load_from_checkpoint(args.checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")

    turns = [
        "hello there, how can I help you?",
        "yes i would like to order some pizza",
        "what toppings would you like?",
        "I want two pizzas one with pepperoni and one with cheese",
    ]
    turns = [
        "hello there, could you tell me about your past job experiences",
        "yes sure my first job was at google as a research engineer and then I went to amazon",
    ]

    words, ps = incremental_turn_prediction(turns, model, tokenizer)
    fig, ax = plot_predictions(words, ps)

    # RANKER

    context = [
        "hello there, could you tell me about your past job experiences",
        "yes sure my first job was at google as a research engineer and then I went to amazon",
    ]
    responses = [
        "could you elaborate on that",
        "That sounds interesting, please tell me more.",
        # "did you enjoy working there?",
        "what made you change direction?",
    ]
    response = get_best_response(context, responses, model, tokenizer)
    print(response)

    context = [
        "Hi there, could you please tell me about yourself?",
        "yes sure! I love to watch movies and enjoy the outdoors. Most of all I am a very happy person",
    ]
    responses = [
        "Tell me more about that",
        "That sounds interesting, please tell me more.",
        "could you elaborate on that",
        "sounds fun how much time do you spend on your hobbies?",
        "interesting",
    ]
    response = get_best_response(context, responses, model, tokenizer)
    print(response)

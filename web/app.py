from argparse import ArgumentParser
import time


import torch
from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.utils import (
    lm_sample,
    TRP,
    predict_trp,
    response_rank,
)
from turngpt.tokenizer import SpokenDialogTokenizer

import flask

app = flask.Flask(__name__)

"""
see ./python_client.py for how to communicate with the server
"""

#############################################################################################3
# API stuff
#############################################################################################3


@app.route("/")
def index():
    return "Index!"


@app.route("/trp", methods=["POST"])
def trp():
    t = time.time()
    data = flask.request.get_json(force=True)
    data = request_to_tokens(data)

    trp = TRP(
        model,
        data["input_ids"],
        data["speaker_ids"],
        sp1_idx=tokenizer.sp1_idx,
        sp2_idx=tokenizer.sp2_idx,
    )[0].tolist()
    # tokens = tokenizer.convert_ids_to_tokens(data["input_ids"][0])
    tokens = tokenizer.ids_to_string(data["input_ids"][0])

    t = round(time.time() - t, 4)
    return {
        "tokens": tokens,
        "trp": trp,
        "time": t,
    }


@app.route("/sample", methods=["POST"])
def sample():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    t = time.time()
    data = flask.request.get_json(force=True)
    data = request_to_tokens(data, add_end_speaker_token=True)

    sample_result = lm_sample(
        model,
        data["input_ids"],
        data["speaker_ids"],
        batch_size=10,
        steps=70,
        topk=5,
        temperature=1.0,
        sample=True,
        stop_at_turn_shift=True,
        max_context=50,
        use_pbar=False,
    )

    # only keep responses
    responses = []
    for output in sample_result["output_ids"]:
        res = output[
            data["input_ids"].shape[-1] : -1
        ]  # omit the input and the last speaker token
        # responses.append(tokenizer.ids_to_string(res))
        responses.append(tokenizer.decode(res))

    responses = response_rank(
        context=data["text"],
        responses=responses,
        model=model,
        tokenizer=tokenizer,
    )
    best_response = responses["responses"][0]
    t = round(time.time() - t, 4)
    return {"response": best_response, "time": t}


@app.route("/prediction", methods=["POST"])
def prediction():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    t = time.time()
    data = flask.request.get_json(force=True)
    data = request_to_tokens(data, add_end_speaker_token=False)

    n_tokens = 3
    horizon = 3
    N = 10
    topk = 5
    temp = 1.0

    # p = predict_trp(
    #     model,
    #     data["input_ids"],
    #     data["speaker_ids"],
    #     n_tokens=n_tokens,
    #     N=N,
    #     topk=topk,
    #     temp=temp,
    #     sp1_idx=tokenizer.sp1_idx,
    #     sp2_idx=tokenizer.sp2_idx,
    # )

    pred = LMUtils.prediction(
        data["text"],
        model=model,
        tokenizer=tokenizer,
        N=N,
        horizon=horizon,
        max_context=70,
        topk=topk,
        temperature=temp,
    )
    del data
    t = round(time.time() - t, 4)
    return {
        "p": pred["p"],
        "predictions": pred["text_predictions"],
        "n_tokens": n_tokens,
        "n": N,
        "time": t,
    }


@app.route("/response_ranking", methods=["POST"])
def response_ranking():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    t = time.time()
    data = flask.request.get_json(force=True)
    responses = response_rank(
        context=data["context"],
        responses=data["responses"],
        model=model,
        tokenizer=tokenizer,
    )
    best_response = responses["responses"][0]
    t = round(time.time() - t, 4)
    return {
        "response": best_response,
        "time": t,
        "tokens": responses["tokens"],
        "eot": responses["eot"],
    }


#############################################################################################3
# Model stuff
#############################################################################################3


class LMUtils(object):
    @staticmethod
    def prediction(
        turn_list,
        model,
        tokenizer,
        N=10,
        horizon=3,
        max_context=200,
        topk=5,
        temperature=1.0,
        sample=True,
    ):
        """"""

        # get input tokens
        input_ids, speaker_ids = tokenizer.turns_to_turngpt_tensors(
            turn_list, add_end_speaker_token=False
        )

        # keys: output_ids, output_speaker, n_steps
        sample_out = lm_sample(
            model,
            input_ids,
            speaker_ids,
            batch_size=N,
            steps=horizon,
            topk=topk,
            temperature=temperature,
            sample=sample,
            stop_at_turn_shift=False,
            max_context=max_context,
            use_pbar=False,
            sp1_idx=tokenizer.sp1_idx,
            sp2_idx=tokenizer.sp2_idx,
        )

        # calculate turn-shift hits
        preds = sample_out["output_ids"][:, -horizon:]
        n_eot = (preds == tokenizer.sp1_idx) + (preds == tokenizer.sp2_idx)
        n_eot = n_eot.sum().item()
        p = n_eot / N

        # get the predicted continuations
        cont = tokenizer.convert_ids_batch_to_string(
            sample_out["output_ids"][:, -horizon:]
        )

        return {"text_predictions": cont, "p": p}


def request_to_tokens(data, add_end_speaker_token=False):
    """
    Assumes data['text'] field exists and is either a single string or a list of strings corresponding to interlocutor
    turns.
    """
    text = data["text"]
    if not isinstance(text, list):
        text = [text]

    input_ids, speaker_ids = tokenizer.turns_to_turngpt_tensors(
        text, add_end_speaker_token=add_end_speaker_token
    )
    data["input_ids"] = input_ids
    data["speaker_ids"] = speaker_ids
    return data


def debug():

    checkpoint = "/home/erik/projects/checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981_V2.ckpt"
    # checkpoint="/home/erik/projects/checkpoints/pretrained/PDEC/checkpoints/epoch=3-val_loss=1.98388.ckpt"
    tokenizer = SpokenDialogTokenizer(pretrained_tokenizer="gpt2")
    model = TurnGPTModel.load_from_checkpoint(checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")

    text = [
        "Hello there, how are you doing today?",
        "I am doing very well I'm",
    ]
    res_prediction = LMUtils.prediction(text, model, tokenizer)


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

    global model
    global tokenizer

    tokenizer = SpokenDialogTokenizer(pretrained_tokenizer="gpt2")
    model = TurnGPTModel.load_from_checkpoint(args.checkpoint)
    if torch.cuda.is_available():
        model = model.to("cuda")

    app.run(port=5001)

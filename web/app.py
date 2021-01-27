from argparse import ArgumentParser
import random

import flask

import torch
from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.utils import lm_sample, TRP, predict_trp, get_best_response
from turngpt.tokenizer import SpokenDialogTokenizer

app = flask.Flask(__name__)

"""
see ./python_client.py for how to communicate with the server
"""


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


@app.route("/")
def index():
    return "Index!"


@app.route("/trp", methods=["POST"])
def trp():
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

    response = {
        "tokens": tokens,
        "trp": trp,
    }
    return flask.jsonify(response)


@app.route("/sample", methods=["POST"])
def sample():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    data = flask.request.get_json(force=True)
    data = request_to_tokens(data, add_end_speaker_token=True)

    sample_result = lm_sample(
        model,
        data["input_ids"],
        data["speaker_ids"],
        batch_size=10,
        steps=100,
        topk=5,
        temperature=1.0,
        sample=True,
        stop_at_turn_shift=True,
        max_context=100,
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

    # choose a random response
    chosen_response = random.choice(responses)
    response = {"response": chosen_response}
    return flask.jsonify(response)


@app.route("/prediction", methods=["POST"])
def prediction():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    data = flask.request.get_json(force=True)
    data = request_to_tokens(data, add_end_speaker_token=False)

    n_tokens = 3
    N = 50
    topk = 5
    temp = 1.0
    p = predict_trp(
        model,
        data["input_ids"],
        data["speaker_ids"],
        n_tokens=n_tokens,
        N=N,
        topk=topk,
        temp=temp,
        sp1_idx=tokenizer.sp1_idx,
        sp2_idx=tokenizer.sp2_idx,
    )
    del data
    response = {"p": p, "n_tokens": n_tokens, "n": N}
    return flask.jsonify(response)


@app.route("/response_ranking", methods=["POST"])
def response_ranking():
    """
    Sample possible responses. Assume that the previous speaker is done.
    """
    data = flask.request.get_json(force=True)
    response = get_best_response(
        context=data["text"],
        responses=data["responses"],
        model=model,
        tokenizer=tokenizer,
    )
    response = {"response": response}
    return flask.jsonify(response)


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

    app.run()

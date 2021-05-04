from argparse import ArgumentParser
import random
import torch

from turngpt.models.pretrained import TurnGPTModel
from turngpt.models.utils import (
    lm_sample,
    TRP,
    predict_trp,
    get_best_response,
    response_rank,
)
from turngpt.tokenizer import SpokenDialogTokenizer


"""
see ./python_client.py for how to communicate with the server
"""


class LMUtils(object):
    """Docstring for LMUtils. """

    def __init__(self, model):
        self.model = model

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

    text = [
        "Hello there, how are you doing today?",
        "I am doing very well thank you, how are you?",
    ]
    res_sample = sample(text, model, tokenizer)
    print("sample: ", res_sample)

    text = [
        "Hello there, how are you doing today?",
        "I am doing very well thank you, how",
    ]
    res_prediction = LMUtils.prediction(text, model, tokenizer)
    print(res_prediction)

    good = True
    if good:
        context = [
            "Hello there, how are you doing today?",
            "I am very happy and energized",
            "that is great tell me more",
            "I just won the lottery in paris i love my life"
            # "I am very happy",
        ]
    else:
        context = [
            "Hello there, how are you doing today?",
            "I am very sad and depressed",
            "oh no what happened",
            "yesterday me and my girlfriend broke up",
        ]
    responses = [
        "great",
        "that is awful",
        "go on",
        "what is wrong",
        "That makes me happy",
        "oh no how can i help",
        "that's great",
    ]
    print("-" * 30)

    res_rank = response_rank(context, responses, model, tokenizer)

    for c in context:
        print(c)
    print("-" * 30)
    for r, score in zip(res_rank["responses"], res_rank["score"]):
        print(r, round(score.item(), 3))

    # print(res_rank))

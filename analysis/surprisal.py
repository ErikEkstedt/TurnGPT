from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from turngpt.TurnGPT import TurnGPT
from turngpt.turngpt_dm import TurnGPTDM
from turngpt.turngpt_utils import get_turns


from ttd.tokenizer_helpers import convert_ids_to_tokens, tokenize_string


"""
TurnGPT (textual)


- Load data without chunks
- get average surprisal over dialog
- plot surprisal over turns (with average)
    - turn-wise
    - N context turns
    - p(x_t | x_t>) for every t in dialog
"""


def load():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser = TurnGPT.add_model_specific_args(parser)
    # temp_args, _ = parser.parse_known_args()
    # print("\nModel Settings")
    # print("--------------")
    # for k, v in vars(temp_args).items():
    #     print(f"{k}: {v}")
    # print("-" * 70)
    # Add data args
    parser = TurnGPTDM.add_data_specific_args(parser, datasets=["dailydialog"])
    args = parser.parse_args()
    args.chunk_size = -1
    args.batch_size = 1
    # ------------------------------------------------------------------
    # Data
    print("\nDataLoader")
    dm = TurnGPTDM(args)
    print("batch size: ", args.batch_size)
    print("num workers: ", args.num_workers)
    print("chunk size: ", args.chunk_size)
    print("explicit_turns: ", args.explicit_turns)
    dm.prepare_data()
    dm.setup("fit")

    model = TurnGPT(
        n_vocab=len(dm.tokenizer),
        pad_idx=dm.tokenizer.pad_token_id,
        sp1_idx=dm.sp1_idx,
        sp2_idx=dm.sp2_idx,
        learning_rate=args.learning_rate,
        proximity_constant=args.proximity_constant,
        args=args,
    )
    print("\n-----", "Model", "-----")
    print("LM:       ", model.lm_model.__class__.__name__)
    if model.acoustic_model is not None:
        print("Acoustic: ", model.acoustic_model.__class__.__name__)
    if model.acoustic_projection is not None:
        print("Acous-proj: ", model.acoustic_projection.__class__.__name__)
    if model.proximity_model is None:
        print("Proximity: None")
    print("n_vocab: ", len(dm.tokenizer))
    print("pad_idx: ", model.pad_idx)
    print("sp1_idx: ", model.sp1_idx)
    print("sp2_idx: ", model.sp2_idx)
    args.checkpoint = (
        # "checkpoints/pretrained/PDEC/checkpoints/epoch=3-val_loss=1.98388.ckpt"
        "checkpoints/pretrained/PDECTWW/checkpoints/epoch=5-val_loss=2.86981.ckpt"
    )
    # Load checkpoint weights
    # strict=False because we might have more parameters in this run...
    if args.checkpoint is not None:
        model = model.load_from_checkpoint(args.checkpoint, strict=False)
        print("Loaded Checkpoint: ", args.checkpoint)
    print()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, dm, args


if __name__ == "__main__":

    model, dm, args = load()

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    end_turn = True
    s = "<speaker1> "
    s += "are you having a good time"
    if end_turn:
        s += " <speaker2>"
    input_ids = tokenize_string(s, dm.tokenizer)
    speaker_ids = torch.tensor([input_ids[0]] * len(input_ids))
    if end_turn:
        speaker_ids[-1] = model.sp2_idx
    input_ids = torch.tensor(input_ids)
    model.block_size = 128
    samples = model.sample(
        input_ids.unsqueeze(0),
        speaker_ids.unsqueeze(0),
        batch_size=20,
        temperature=1.1,
        steps=5000,
        # steps=0,
    )
    ranked = []
    for ids, sp in zip(samples["output_ids"], samples["output_speaker"]):
        inp, sp = ids.unsqueeze(0).to(model.device), sp.unsqueeze(0).to(model.device)
        lab = inp[:, 1:]
        inp = inp[:, :-1]
        sp = sp[:, :-1]
        o = model(inp, sp)
        l = loss_fn(o["logits"][0], lab[0])
        if len(l) > len(input_ids):
            l = l[len(input_ids) :].mean().item()
        else:
            l = -1.0
        tokens = " ".join(convert_ids_to_tokens(ids, dm.tokenizer)[1:-1])
        ranked.append({"loss": l, "tokens": tokens})
    ranked.sort(key=lambda x: x["loss"])
    for r in ranked:
        tokens = r["tokens"]
        l = r["loss"]
        print(str(round(l, 2)).zfill(3), "|", tokens)

    input_ids = samples["output_ids"]

    loader = dm.val_dataloader()

    for batch in loader:
        input_ids, speaker_ids, labels, *_ = model.shared_step(batch)
        with torch.no_grad():
            out = model(input_ids.to(model.device), speaker_ids.to(model.device))
            loss = loss_fn(out["logits"][0], labels[0].to(model.device)).cpu()
        ce_loss = loss.mean()
        print("ce loss: ", ce_loss.item())

    batch = next(iter(loader))

    N = 1
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for batch in loader:
        input_ids, speaker_ids, labels, *_ = model.shared_step(batch)
        with torch.no_grad():
            out = model(input_ids.to(model.device), speaker_ids.to(model.device))
            loss = loss_fn(out["logits"][0], labels[0].to(model.device)).cpu()
        avg_ce_loss = loss.mean()

        tokens = convert_ids_to_tokens(labels, dm.tokenizer)[0]
        turns = get_turns(input_ids, model.sp1_idx, model.sp2_idx)[0]
        for t in range(len(turns) - 1):
            ax.cla()
            c_start, c_end = turns[t, 0], turns[t + (N - 1), 1] - 1
            c_len = c_end - c_start
            u_start, u_end = turns[t + N, 0] - 1, turns[t + N, 1]
            u_len = u_end - u_start

            # Context
            context_loss = loss[c_start:c_end]
            x_c = torch.arange(c_len)
            ax.plot(x_c, context_loss, label="context")
            ax.hlines(
                context_loss.mean().item(),
                0,
                len(x_c),
                linestyle="dashed",
                color="b",
                alpha=0.5,
                label="context avg",
            )

            # Current Utterance
            utter_loss = loss[u_start:u_end]
            x_u = torch.arange(c_len, c_len + u_len)
            ax.plot(x_u, utter_loss, label="current")
            ax.hlines(
                utter_loss.mean().item(),
                len(x_c),
                len(x_c) + len(x_u),
                linestyle="dashed",
                color="y",
                alpha=0.3,
                label="current avg",
            )

            # general
            tmp_tokens = tokens[c_start:u_end]
            ax.hlines(
                avg_ce_loss.item(),
                0,
                len(tmp_tokens),
                linestyle="dashed",
                color="k",
                alpha=0.5,
                linewidth=0.8,
                label="dialog avg",
            )
            ax.set_xticks(torch.arange(len(tmp_tokens)))
            ax.set_xticklabels(tmp_tokens, rotation=65)
            ax.legend()
            plt.tight_layout()
            plt.pause(0.0001)
            input()

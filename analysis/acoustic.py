from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from turngpt.TurnGPT import TurnGPT
from turngpt.acousticDM import AcousticGPTDM
from turngpt.turngpt_utils import get_turns, get_speaker_shift_indices

from ttd.tokenizer_helpers import convert_ids_to_tokens, tokenize_string


def load(checkpoint=None, datasets=["switchboard"], post_silence=False):
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=checkpoint)
    parser = TurnGPT.add_model_specific_args(parser)
    parser = AcousticGPTDM.add_data_specific_args(
        parser, datasets=datasets, post_silence=post_silence
    )
    args = parser.parse_args()
    args.chunk_size = 128
    args.batch_size = 1
    args.prosody = True
    # ------------------------------------------------------------------
    # Data
    print("\nDataLoader")
    dm = AcousticGPTDM(args)
    print("batch size: ", args.batch_size)
    print("num workers: ", args.num_workers)
    print("chunk size: ", args.chunk_size)
    print("explicit_turns: ", args.explicit_turns)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    if args.checkpoint is None:
        model = TurnGPT(
            n_vocab=len(dm.tokenizer),
            pad_idx=dm.tokenizer.pad_token_id,
            sp1_idx=dm.sp1_idx,
            sp2_idx=dm.sp2_idx,
            learning_rate=args.learning_rate,
            proximity_constant=args.proximity_constant,
            args=args,
        )
    else:
        model = TurnGPT.load_from_checkpoint(args.checkpoint)
        print("Loaded: ", args.checkpoint)

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

    if torch.cuda.is_available():
        model.to("cuda")
    return model, dm, args


def plot(
    inp_tokens,
    loss,
    trp,
    prox_prob1,
    prox_prob3,
    attn_mix,
    attn_feat,
    sp_idx,
    post_silence=None,
    show_pitch=True,
    show_voiced=False,
    show_zcr=False,
    show_rms=False,
    show_words=False,
    ax=None,
    fig=None,
    plot=True,
):
    if ax is None:
        if post_silence is not None:
            fig, ax = plt.subplots(5, 1, figsize=(15, 9), sharex=True)
        else:
            fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    else:
        for a in ax:
            a.cla()

    feats = ["pitch", "voiced", "zcr", "rms"]

    ax[0].plot(loss, label="NLL", alpha=0.6)
    ax[0].legend()
    ax[1].plot(prox_prob1, label="prox 1", alpha=0.6)
    ax[1].plot(prox_prob3, label="prox 3", alpha=0.3, linewidth=1.5)
    ax[1].plot(trp, "k--", label="lm trp", alpha=0.3, linewidth=0.8)
    ax[1].vlines(sp_idx, 0, 1, "k", linewidth=2, alpha=0.2)
    ax[1].legend()

    if show_pitch:
        ax[2].plot(attn_feat[:, 0], alpha=0.4, label="pitch")
    if show_voiced:
        ax[2].plot(attn_feat[:, 1], alpha=0.4, label="voiced")
    if show_zcr:
        ax[2].plot(attn_feat[:, 2], alpha=0.4, label="zcr")
    if show_rms:
        ax[2].plot(attn_feat[:, 3], alpha=0.4, label="rms")
    ax[2].legend()
    ax[2].vlines(sp_idx, 0, 1, "k", linewidth=2, alpha=0.2)
    ax[2].set_ylim([0, 1.0])

    ax[3].plot(attn_mix[:, 1], alpha=0.4, label="feats")
    if show_words:
        ax[3].plot(attn_mix[:, 0], alpha=0.4, label="words")
    ax[3].vlines(sp_idx, 0, 1, "k", linewidth=2, alpha=0.2)
    ax[3].hlines(
        0.5, 0, len(inp_tokens), "k", linewidth=1, alpha=0.2, linestyle="dashed"
    )
    ax[3].legend()
    ax[3].set_ylim([0, 1.0])

    if post_silence is not None:
        ax[4].plot(post_silence, label="post_silence")
        ax[4].vlines(sp_idx, 0, 1, "k", linewidth=2, alpha=0.2)
        ax[4].hlines(
            0, 0, len(inp_tokens), "k", linewidth=1, alpha=0.2, linestyle="dashed"
        )
        ax[4].legend()

    ax[-1].set_xticks(torch.arange(len(inp_tokens)))
    ax[-1].set_xticklabels(inp_tokens, rotation=65)
    ax[-1].set_xlim([0, len(inp_tokens)])

    plt.tight_layout()
    if plot:
        plt.pause(0.01)
    return fig, ax


if __name__ == "__main__":

    # checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained_A-conv_P-transformer/version_0/checkpoints/epoch=1-val_loss=5.31956.ckpt"
    # checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained_A-conv_P-transformer/version_5/checkpoints/epoch=2-val_loss=4.68522.ckpt"
    # checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained_A-conv_P-transformer/version_7/checkpoints/epoch=1-val_loss=5.59143.ckpt"
    checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained_A-conv_P-transformer/version_8/checkpoints/epoch=0-val_loss=3.68695.ckpt"
    checkpoint = "TurnGPT/turngpt/runs/TurnGPTpretrained_A-conv_P-transformer/version_8/checkpoints/epoch=1-val_loss=3.68129.ckpt"
    checkpoint = "checkpoints/pros/epoch=1-val_loss=3.85717.ckpt"
    model, dm, args = load(checkpoint, datasets=["switchboard"], post_silence=True)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    # TODO: is the offset correct?
    # why is not speaker1/2 post_silence the same as its neighbours?
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    b = model.shared_step(batch)

    p = b["spf"][0]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for tp in p:
        ax.cla()
        ax.plot(tp[:, 0].unsqueeze(1), label="pitch")
        ax.plot(tp[:, 1].unsqueeze(1), label="voiced")
        ax.plot(tp[:, 2].unsqueeze(1), label="zcr")
        ax.plot(tp[:, 3].unsqueeze(1), label="rms")
        ax.legend()
        plt.pause(0.001)
        input()

    # for i in range(b["input_ids"].shape[1]):
    #     print(b["input_ids"][0, i], b["post_silence"][0, i])
    #     input()
    # ps = b["post_silence"][:, 1:, 0] - b["post_silence"][:, :-1, 1]
    # sp_b, sp_idx = get_speaker_shift_indices(
    #     b["input_ids"], model.sp1_idx, model.sp2_idx
    # )
    # pre_shift = ps[sp_b, sp_idx - 1]
    # print("-" * 30)
    # print(b["post_silence"][sp_b, sp_idx - 1][..., -1])
    # print(b["post_silence"][sp_b, sp_idx][..., 0])
    # print(b["input_ids"][sp_b, sp_idx])
    # print(pre_shift)
    # input()

    if False:
        use_ps = False
        if use_ps:
            fig, ax = plt.subplots(5, 1, figsize=(15, 9), sharex=True)
        else:
            fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True)

        for batch in loader:
            b = model.shared_step(batch)
            # input_ids, speaker_ids, labels, audio, features = model.shared_step(batch)
            input_ids = b["input_ids"]
            speaker_ids = b["speaker_ids"]
            labels = b["labels"]
            features = b["spf"]
            with torch.no_grad():
                out = model(
                    input_ids.to(model.device),
                    speaker_ids.to(model.device),
                    waveform=None,
                    spf=features.to(model.device),
                    output_attentions=True,
                )
                trp = F.softmax(out["logits"], dim=-1)
                trp = torch.stack(
                    (trp[..., model.sp1_idx], trp[..., model.sp2_idx]), dim=-1
                )
                trp, _ = trp.max(dim=-1)
                trp = trp[0].cpu()
                loss = loss_fn(out["logits"][0], labels[0].to(model.device)).cpu()
            # tokens
            inp_tokens = convert_ids_to_tokens(input_ids[0], dm.tokenizer)
            _, sp_idx = get_speaker_shift_indices(
                input_ids, model.sp1_idx, model.sp2_idx
            )
            # data
            prox_prob1 = torch.sigmoid(out["proximity_logits"][0, :, 0]).cpu()
            prox_prob3 = torch.sigmoid(out["proximity_logits"][0, :, 1]).cpu()
            # attention
            attn_feat = out["attn_feat"][0].cpu()
            attn_mix = out["attn_mix"][0].cpu()
            tmp_post_silence = None
            if use_ps:
                ps = b["post_silence"][:, 1:, 0] - b["post_silence"][:, :-1, 1]
                sp_b, sp_idx = get_speaker_shift_indices(
                    b["input_ids"], model.sp1_idx, model.sp2_idx
                )
                ps[sp_b, sp_idx] = ps[sp_b, sp_idx + 1]
                tmp_post_silence = ps[0]
            fig, ax = plot(
                inp_tokens,
                loss,
                trp,
                prox_prob1,
                prox_prob3,
                attn_mix,
                attn_feat,
                sp_idx,
                show_voiced=True,
                show_zcr=True,
                show_rms=True,
                post_silence=tmp_post_silence,
                ax=ax,
                fig=fig,
                plot=True,
            )
            input()

        b["post_silence"][0, :10]
        b["input_ids"][0, :10]

        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        ax.plot(prox_prob1, label="prox 1", alpha=0.6)
        ax.plot(prox_prob3, label="prox 3", alpha=0.3, linewidth=1.5)
        ax.plot(trp, "k--", label="lm trp", alpha=0.3, linewidth=0.8)
        ax.set_xticks(torch.arange(len(inp_tokens)))
        ax.set_xticklabels(inp_tokens, rotation=65)
        ax.set_xlim([0, len(inp_tokens)])
        ax.legend()
        plt.tight_layout()
        plt.pause(0.1)

        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        # ax[0].imshow(attn_feat.T, origin='lower', aspect='auto')
        ax[0].imshow(attn_feat.T, aspect="auto")
        ax[0].set_yticks(range(4))
        ax[0].set_yticklabels(["pitch", "voiced", "zcr", "rms"])
        # ax[1].imshow(attn_mix.T, origin='lower', aspect='auto')
        ax[1].imshow(attn_mix.T, aspect="auto")
        ax[1].set_yticks(range(2))
        ax[1].set_yticklabels(["word", "audio"])
        ax[1].set_xticks(torch.arange(len(inp_tokens)))
        ax[1].set_xticklabels(inp_tokens, rotation=65)
        ax[1].set_xlim([0, len(inp_tokens)])
        plt.tight_layout()
        plt.pause(0.01)

        batch = next(iter(loader))
        input_ids, speaker_ids, labels, audio, features = model.shared_step(batch)
        with torch.no_grad():
            out = model(
                input_ids.to(model.device),
                speaker_ids.to(model.device),
                waveform=None,
                spf=features.to(model.device),
                output_attentions=True,
            )
            trp = F.softmax(out["logits"], dim=-1)
            trp = torch.stack(
                (trp[..., model.sp1_idx], trp[..., model.sp2_idx]), dim=-1
            )
            trp, _ = trp.max(dim=-1)
            trp = trp[0].cpu()
            loss = loss_fn(out["logits"][0], labels[0].to(model.device)).cpu()
        # data
        prox_prob1 = torch.sigmoid(out["proximity_logits"][0, :, 0]).cpu()
        prox_prob3 = torch.sigmoid(out["proximity_logits"][0, :, 1]).cpu()

        # tokens
        inp_tokens = convert_ids_to_tokens(input_ids[0], dm.tokenizer)
        _, sp_idx = get_speaker_shift_indices(input_ids, model.sp1_idx, model.sp2_idx)

        plot

        _, ii = get_speaker_shift_indices(labels, model.sp1_idx, model.sp2_idx)

        pos_prob = prox_prob1[ii]
        pos_trp = trp[ii]
        neg_prob = prox_prob1[ii]
        neg_trp = trp[ii]

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def trp_from_logits(logits, sp1_idx, sp2_idx):
    probs = F.softmax(logits, dim=-1)
    trp, _ = probs[:, :, (sp1_idx, sp2_idx)].max(dim=-1)
    return trp


def TRP(
    model,
    input_ids,
    speaker_ids=None,
    sp1_idx=50257,
    sp2_idx=50258,
):
    if speaker_ids is not None:
        speaker_ids = speaker_ids.to(model.device)
    out = model(input_ids.to(model.device), speaker_ids=speaker_ids)
    return trp_from_logits(out["logits"], sp1_idx, sp2_idx)


def get_response_batch(context, responses, tokenizer):
    input_ids, speaker_ids = tokenizer.turns_to_turngpt_tensors(
        context, add_end_speaker_token=True
    )
    last_speaker = speaker_ids[0, -1]
    new_speaker = (
        tokenizer.sp1_idx if last_speaker == tokenizer.sp1_idx else tokenizer.sp1_idx
    )
    new_speaker = torch.tensor((new_speaker,))
    context_len = input_ids.shape[-1]

    r, l, sp = [], [], []
    for res in responses:
        # tmp = torch.cat((new_speaker, torch.tensor(tokenizer.encode(res))))
        tmp = torch.tensor(tokenizer.encode(res))
        r.append(tmp)
        l.append(context_len + len(tmp) - 1)
    r = pad_sequence(
        r, batch_first=True, padding_value=tokenizer._tokenizer.pad_token_id
    )

    tmp_sp = new_speaker.repeat(r.shape)
    speaker_ids = torch.cat([speaker_ids] * len(r))
    speaker_ids = torch.cat([speaker_ids, tmp_sp], dim=-1)

    input_ids = torch.cat([input_ids] * len(r))
    input_ids = torch.cat((input_ids, r), dim=-1)

    return input_ids, speaker_ids, l, context_len


def get_best_response(context, responses, model, tokenizer):
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    input_ids, speaker_ids, response_ends, context_len = get_response_batch(
        context, responses, tokenizer
    )

    out = model(input_ids.to(model.device), speaker_ids.to(model.device))

    labels = input_ids[:, 1:].contiguous()
    logits = out["logits"][:, :-1].contiguous()

    ce = loss_fn(
        logits.view(-1, logits.size(-1)), labels.view(-1).to(model.device)
    ).reshape(logits.shape[:-1])
    # context_ce = ce[:, : context_len - 1].mean(dim=-1)

    response_ce = []
    for i, end in enumerate(response_ends):
        res_ce = ce[i, context_len:end].mean().item()
        response_ce.append(res_ce)

    response_ce = torch.tensor(response_ce)
    res_ce, perm_idx = response_ce.sort(descending=False)
    best_response = perm_idx[0]
    response = responses[best_response]
    return response


def predict_trp(
    model,
    input_ids,
    speaker_ids,
    N,
    n_tokens,
    topk=5,
    temp=1.0,
    sp1_idx=50257,
    sp2_idx=50258,
):
    sample_result = lm_sample(
        model,
        input_ids,
        speaker_ids,
        batch_size=N,
        steps=n_tokens,
        topk=topk,
        temperature=temp,
        sample=True,
        stop_at_turn_shift=False,
        max_context=128,
        use_pbar=False,
    )
    preds = sample_result["output_ids"][:, -n_tokens:]
    n_eot = (preds == sp1_idx) + (preds == sp2_idx)
    n_eot = n_eot.sum().item()
    p = n_eot / N
    return p


@torch.no_grad()
def lm_sample(
    model,
    input_ids,
    speaker_ids,
    batch_size=-1,
    steps=50,
    topk=5,
    temperature=1.0,
    sample=True,
    stop_at_turn_shift=True,
    max_context=100,
    use_pbar=False,
    sp1_idx=50257,
    sp2_idx=50258,
):
    def get_next_speaker_ids(next_idx, speaker_ids, sp1_idx, sp2_idx):
        """
        If the model outputs a speaker shift we must change the corresponding speaker tokens for the next step
        """
        next_speaker_ids = torch.zeros_like(next_idx)
        for i, (next_idx, last_speaker) in enumerate(zip(next_idx, speaker_ids[:, -1])):
            if next_idx == sp1_idx or next_idx == sp2_idx:
                next_speaker = next_idx
            else:
                next_speaker = last_speaker
            next_speaker_ids[i] = next_speaker
        return next_speaker_ids

    def pop_complete_samples(next_idx, step, sample_data, sp1_idx, sp2_idx):
        """Check which samples have produced a speaker shift and remove them from being processed in the next step"""
        sp1_batch, _ = torch.where(next_idx == sp1_idx)
        sp2_batch, _ = torch.where(next_idx == sp2_idx)
        sp_batch = torch.cat((sp1_batch, sp2_batch))
        if len(sp_batch) > 0:
            # Omit all samples which produced a speaker shift and add them to the output_samples
            for i, b in enumerate(sp_batch):
                sample_data["n_steps"].append(step + 1)
                sample_data["completed_ids"].append(sample_data["all_input_ids"][b])
                sample_data["completed_speaker"].append(
                    sample_data["all_speaker_ids"][b]
                )

            # Keep the remaining batches such that they may be processed in the next step
            keep_batches = [
                ind
                for ind in range(sample_data["all_input_ids"].shape[0])
                if ind not in sp_batch
            ]
            sample_data["input_ids"] = sample_data["input_ids"][keep_batches]
            sample_data["speaker_ids"] = sample_data["speaker_ids"][keep_batches]
            sample_data["all_input_ids"] = sample_data["all_input_ids"][keep_batches]
            sample_data["all_speaker_ids"] = sample_data["all_speaker_ids"][
                keep_batches
            ]
        return sample_data

    # Get more samples by repeating the input by 'batch_size'
    # e.g. (1, 20) -> (10, 20) with batch_size=10
    # or   (5, 20) -> (50, 20) with batch_size=10
    if batch_size > 1:
        input_ids = torch.cat([input_ids] * batch_size)
        if speaker_ids is not None:
            speaker_ids = torch.cat([speaker_ids] * batch_size)

    sample_data = {
        "input_ids": input_ids,  # maximum of context size, the tensor used to sample
        "speaker_ids": speaker_ids,  # might be None
        "all_input_ids": input_ids.clone().cpu(),  # stores all values even longer than context
        # "avg_log_probs": [],  # TODO
    }

    if speaker_ids is not None:
        sample_data["all_speaker_ids"] = speaker_ids.clone().cpu()

    if stop_at_turn_shift:
        sample_data["n_steps"] = []  # store the number of steps for each sample
        sample_data["completed_ids"] = []  # stores the completed inputs token samples
        sample_data["completed_speaker"] = []  # stores the completed speaker tokens

    # Move data to correct device
    sample_data["input_ids"] = sample_data["input_ids"].to(model.device)
    sample_data["speaker_ids"] = sample_data["speaker_ids"].to(model.device)

    # Use a progress bar
    if use_pbar:
        pbar = tqdm(range(steps), desc="Sampling")
    else:
        pbar = range(steps)

    # Sample loop
    for step in pbar:
        # Only process the maximum context length
        sample_data["input_ids"] = sample_data["input_ids"][:, -max_context:]
        if sample_data["speaker_ids"] is not None:
            sample_data["speaker_ids"] = sample_data["speaker_ids"][:, -max_context:]

        ###############################################################################
        # Forward pass
        # TODO: use past in forward pass
        out = model(sample_data["input_ids"], speaker_ids=sample_data["speaker_ids"])
        lm_logits = out["logits"][:, -1, :] / temperature  # Prediction + scale w/ temp

        # only the topk entries (only useful when sampling... greedy approach don't care)
        if topk is not None and sample:
            lm_logits, topk_idx = lm_logits.topk(k=topk, dim=-1)
        probs = F.softmax(lm_logits, dim=-1)

        ###############################################################################
        # Sample or greedy
        if sample:
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # TODO
            # next_log_prob = probs[next_idx].log()
            if topk is not None:
                # remap back to orignial indices
                next_idx = topk_idx[
                    torch.arange(next_idx.shape[0]), next_idx[:, 0]
                ].unsqueeze(-1)
        else:  # GREEDY
            probs, sorted_ids = probs.sort(descending=True)
            next_idx = sorted_ids[:, :1]
            # TODO
            # next_log_prob = probs[next_idx].log()

        ###############################################################################
        # Fix speaker_ids. If there has been a change in speaker we must provide the correct speaker_tokens
        if speaker_ids is not None:
            next_speaker_ids = get_next_speaker_ids(
                next_idx, sample_data["speaker_ids"], sp1_idx, sp2_idx
            )

        ###############################################################################
        # Append to the sequence and continue
        sample_data["input_ids"] = torch.cat(
            (sample_data["input_ids"], next_idx), dim=-1
        )
        sample_data["all_input_ids"] = torch.cat(
            (sample_data["all_input_ids"], next_idx.cpu()), dim=-1
        )

        if sample_data["speaker_ids"] is not None:
            sample_data["speaker_ids"] = torch.cat(
                (sample_data["speaker_ids"], next_speaker_ids), dim=-1
            )
            sample_data["all_speaker_ids"] = torch.cat(
                (sample_data["all_speaker_ids"], next_speaker_ids.cpu()), dim=-1
            )

        ###############################################################################
        if stop_at_turn_shift:
            sample_data = pop_complete_samples(
                next_idx, step, sample_data, sp1_idx, sp2_idx
            )
            # If all sequences have been processed until a speaker shift we return the results
            if len(sample_data["all_input_ids"]) == 0:
                return {
                    "output_ids": sample_data["completed_ids"],
                    "output_speaker": sample_data["completed_speaker"],
                    "n_steps": sample_data["n_steps"],
                }

    return {
        "output_ids": sample_data["all_input_ids"],
        "output_speaker": sample_data["all_speaker_ids"],
        "n_steps": [step + 1] * sample_data["all_input_ids"].shape[0],
    }

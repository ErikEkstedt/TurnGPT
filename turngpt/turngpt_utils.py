import torch
from ttd.tokenizer_helpers import remove_punctuation_capitalization


def onehot_speaker_shift_labels(input_ids, sp1_idx, sp2_idx):
    lab1 = torch.where(input_ids == sp1_idx)
    lab2 = torch.where(input_ids == sp2_idx)
    labels = torch.zeros_like(input_ids).float()
    labels[lab1] = 1.0
    labels[lab2] = 1.0
    return labels


def get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx):
    inp = input_ids.clone()
    inp[inp == sp2_idx] = sp1_idx
    sp_b, sp_inds = torch.where(inp == sp1_idx)  # all speaker 1 tokens
    return (sp_b, sp_inds)


def get_turn_shift_indices(input_ids, sp1_idx, sp2_idx):
    ts_bs, ts_inds = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)
    ts_inds = ts_inds - 1  # turn-shift are
    ts_bs = ts_bs[ts_inds != -1]
    ts_inds = ts_inds[ts_inds != -1]
    return (ts_bs, ts_inds)


def get_turns(input_ids, sp1_idx, sp2_idx):
    assert input_ids.ndim == 2
    sp_b, sp_inds = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)
    turns = []
    for b in range(input_ids.shape[0]):
        turns.append(sp_inds[sp_b == b].unfold(0, 2, 1))
    return turns


def get_positive_and_negative_indices(input_ids, sp1_idx, sp2_idx, pad_idx):
    """
    Finds positive and negative indices for turn-shifts.

    * Positive turn-shifts are the indices prior to a <speaker1/2> token
    * Negative turn-shifts are all other indices (except pad_tokens)

    Returns:
        turn_shift_indices:     tuple, (batch, inds) e.g.  input_ids[turn_shift_indices]
        non_turn_shift_indices: tuple, (batch, inds) e.g.  input_ids[non_turn_shift_indices]
    """
    (ts_bs, ts_inds) = get_turn_shift_indices(input_ids, sp1_idx, sp2_idx)
    bp, indp = torch.where(input_ids != pad_idx)  # all valid places

    # TODO:
    # Remove the speaker-id tokens from negatives?

    neg_bs, neg_inds = [], []
    for i in bp.unique():
        neg_ind = indp[bp == i]  # valid indices (not pad) # [1:]  # omit 0
        ts = ts_inds[ts_bs == i]  # turn-shifts in batch i
        neg_ind[ts] = -1  # mark these
        neg_ind = neg_ind[neg_ind != -1]
        neg_bs.append(torch.ones_like(neg_ind) * i)
        neg_inds.append(neg_ind)

    neg_bs = torch.cat(neg_bs)
    neg_inds = torch.cat(neg_inds)
    return (ts_bs, ts_inds), (neg_bs, neg_inds)


def find_turn_context(focus_index, turns, n_context):
    """
    Finds in which turn `focus_index` is and returns the relevant turn
    along with the context defined by `n_context`
    """
    for n_turn, (s, e) in enumerate(turns):
        if e - 1 == focus_index:
            break
    return turns[n_turn - n_context : n_turn + 1]


def turns_to_turngpt_tensors(turns, tokenizer, explicit_turn_shift=True):
    assert isinstance(turns, list), "turns must be a list of strings"
    turns = [remove_punctuation_capitalization(text) for text in turns]
    sp1_idx, sp2_idx = (
        tokenizer.convert_tokens_to_ids("<speaker1>"),
        tokenizer.convert_tokens_to_ids("<speaker2>"),
    )
    input_ids, speaker_ids = [], []
    for i, t in enumerate(turns):
        toks = tokenizer.encode(t)
        if i % 2 == 0:
            cur_speaker = sp1_idx
        else:
            cur_speaker = sp2_idx
        if explicit_turn_shift:
            input_ids.append(cur_speaker)
            speaker_ids.append(cur_speaker)
        input_ids += toks
        speaker_ids += [cur_speaker] * len(toks)
    assert len(speaker_ids) == len(input_ids)
    return (
        torch.tensor(input_ids).unsqueeze(0),
        torch.tensor(speaker_ids).unsqueeze(0),
    )


def get_focus_indices(trp, input_ids, prob_thresh, n_context, sp1_idx, sp2_idx):
    """get_focus_indices.

    Gets focus-indices where the model assigns a likelihood over `prob_thresh` over locations prior to actual
    turn-shifts. Makes sure that there is `n_context` turns prior to the current utterance. Returns the batch and
    sequence indices.

    :param input_ids:       torch.tensor, input tokens
    :param speaker_ids:     torch.tensor, speaker tokens
    :param prob_thresh:     float, probability threshold that defines "likely" turn-shifts
    :param n_context:       int, number of context turns prior to the utterance where a turn-shift is likely

    Returns:
        focus_bs:           torch.tensor, batch of calculated focus indices
        focus_inds:         torch.tensor, index of calculated focus indices
    """

    # Find prediction where actual turn-shifts are present in the data,
    # keep only those predictions over a certain probaility threshold.
    # i.e. moments where there should be a turn-shift prediction and
    # the model assign a high likelihood for that being the case.
    ts_bs, ts_inds = get_turn_shift_indices(input_ids, sp1_idx=sp1_idx, sp2_idx=sp2_idx)
    positive_guesses = trp[(ts_bs, ts_inds)].cpu()
    over_thresh = torch.where(positive_guesses >= prob_thresh)
    possible_focus_bs = ts_bs[over_thresh]
    possible_focus_inds = ts_inds[over_thresh]

    # Keep the likely true-positives that have sufficient context
    focus_bs = []
    focus_inds = []
    turns = get_turns(input_ids, sp1_idx, sp2_idx)
    for b, t in enumerate(turns):
        if len(t) > n_context:
            min_ind = t[n_context][0].item()
            possible_focus = possible_focus_inds[possible_focus_bs == b]
            tmp_focus_inds = possible_focus[possible_focus > min_ind]
            focus_bs.append(torch.ones_like(tmp_focus_inds).fill_(b))
            focus_inds.append(tmp_focus_inds)
    if len(focus_bs) > 0:
        focus_bs = torch.cat(focus_bs)
        focus_inds = torch.cat(focus_inds)
    return focus_bs, focus_inds


def batch_to_context_ablation_batch(
    input_ids,
    speaker_ids,
    n_context,
    sp1_idx,
    sp2_idx,
    omit_speaker_toks_as_negatives=False,
    sort=True,
):
    """
    Create a large batch with all context sizes and keep track of the context used
    and the indices for positive/negative places (used for score)
    """
    context_ids = []
    context_speaker = []
    total_lens = []
    pos_context = []
    pos_indices = []
    neg_context = []
    neg_indices = []
    batch_turns = get_turns(input_ids, sp1_idx, sp2_idx)
    for b, turns in enumerate(batch_turns):
        if len(turns) > n_context:
            valid_turns = turns.unfold(0, size=n_context + 1, step=1)
            for vt in valid_turns:
                # vt[start/end, n_turn]
                focus_start = vt[0, -1]
                end = vt[1, -1]
                for i in range(n_context + 1):
                    start = vt[0, i]
                    tmp_focus_start = focus_start - start
                    tmp_focus_end = end - start - 1
                    context = n_context - i
                    context_ids.append(
                        input_ids[b, start : end + 1]
                    )  # will not include last sp_token
                    context_speaker.append(speaker_ids[b, start : end + 1])
                    pos_context.append(context)
                    total_lens.append(len(context_ids[-1]))
                    # pos/neg target indices
                    pos_indices.append(tmp_focus_end)
                    if omit_speaker_toks_as_negatives:
                        neg = torch.arange(tmp_focus_start + 1, tmp_focus_end)
                        if len(neg) > 0:
                            neg_indices.append(neg)
                            neg_context.append(
                                torch.ones_like(neg_indices[-1]) * context
                            )
                    else:
                        neg_indices.append(torch.arange(tmp_focus_start, tmp_focus_end))
                        neg_context.append(torch.ones_like(neg_indices[-1]) * context)
    # Create context batch
    # we sort the datapoints for more efficient forward pass
    if sort:
        total_lens, perm_idx = torch.tensor(total_lens).sort(descending=True)
        context_ids = [context_ids[i] for i in perm_idx]
        context_speaker = [context_speaker[i] for i in perm_idx]
        pos_context = [pos_context[i] for i in perm_idx]
        neg_context = [neg_context[i] for i in perm_idx]
        pos_indices = [pos_indices[i] for i in perm_idx]
        neg_indices = [neg_indices[i] for i in perm_idx]
    return {
        "context_ids": context_ids,
        "context_speaker": context_speaker,
        "pos_context": pos_context,
        "neg_context": neg_context,
        "pos_indices": pos_indices,
        "neg_indices": neg_indices,
    }

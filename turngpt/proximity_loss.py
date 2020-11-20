import torch
import torch.nn.functional as F


def proximity_loss(logits, labels, N, sp1_idx, sp2_idx=None):
    """
    Labels are token indices shifted to the left.
    Create onehot-vector representing turn-shifts in label.
    Unfold the tensor to look ahead N tokens.
    Sum the onehot values in the unfolded window dim and check if any 1s are present.
    A zero value in the labels indicates no proximity of turn-shift.

    :logits:        torch.Tensor (B, N, 2)
    :labels:        torch.Tensor (B, N)
    :N:             int, n_tokens of horizon
    :sp1_idx:       int, speaker 1 token index
    :sp2_idx:       int, speaker 2 token index
    """
    proximity_labels = labels == sp1_idx
    if sp1_idx != sp2_idx:
        proximity_labels += labels == sp2_idx
    proximity_labels = proximity_labels.unfold(dimension=1, step=1, size=N).sum(dim=-1)
    proximity_labels = (proximity_labels > 0).long().detach()

    logits = logits[:, : proximity_labels.shape[1]].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        proximity_labels.view(-1),
    )

import torch
import numpy as np

from turngpt.turngpt_utils import get_positive_and_negative_indices


class ClassificationLabelTransform(object):
    def __init__(
        self,
        ratio=1,
        sp1_idx=50257,
        sp2_idx=50258,
        pad_idx=50256,
        max_batch_size=256,
        unigram=True,
    ):
        self.ratio = ratio
        self.sp1_idx = sp1_idx
        self.sp2_idx = sp2_idx
        self.pad_idx = pad_idx
        self.unigram = unigram

        self.max_batch_size = max_batch_size

    def _unigram(self, x, input_ids):
        pos, neg = get_positive_and_negative_indices(
            input_ids, self.sp1_idx, self.sp2_idx, self.pad_idx
        )

        # positive
        n_pos = len(pos[0])
        pos_x = x[pos]

        # negative
        n_neg = len(neg[0])
        N = int(n_pos / self.ratio)
        neg_idx = torch.from_numpy(np.random.choice(n_neg, N)).long()
        neg_x = x[neg][neg_idx]

        pos_inp = input_ids[pos]
        neg_inp = input_ids[neg][neg_idx]
        inp = torch.cat((pos_inp, neg_inp))

        x = torch.cat((pos_x, neg_x))
        y = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        y[:n_pos] = 1
        return x, y, inp

    def onehot_speaker_shift(self, x, input_ids):
        sp = input_ids == self.sp1_idx
        sp += input_ids == self.sp2_idx
        sp = torch.cat(
            (sp[:, 1:], torch.zeros(sp.shape[0], 1).to(input_ids.device)), dim=-1
        )
        return x, sp, input_ids

    def __call__(self, x, input_ids):

        if self.unigram:
            return self._unigram(x, input_ids)
        else:
            return self.onehot_speaker_shift(x, input_ids)


if __name__ == "__main__":

    from argparse import ArgumentParser
    from turngpt.acousticDM import AudioDM
    from ttd.tokenizer_helpers import convert_ids_to_tokens
    from turngpt.turngpt_utils import get_speaker_shift_indices
    from turngpt.models import gradient_check_batch, gradient_check_word_time
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser = AudioDM.add_data_specific_args(
        parser,
        datasets=["maptask"],
        f0=True,
        waveform=True,
        f0_normalize=True,
        f0_interpolate=True,
        rms=True,
        log_rms=True,
    )
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    args.batch_size = 16
    dm = AudioDM(args)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()

    batch = next(iter(loader))

    batch_transform = ClassificationLabelTransform(
        ratio=1.0,
        sp1_idx=dm.sp1_idx,
        sp2_idx=dm.sp2_idx,
        pad_idx=dm.pad_idx,
        unigram=False,
    )

    input_ids = batch["input_ids"]
    x = torch.stack((batch["f0"], batch["rms"]), dim=-1)
    print("pros: ", x.shape)
    x, y, inp = batch_transform(x, input_ids)
    print("x: ", x.shape, x.device, x.dtype)
    print("y: ", y.shape, y.device, y.dtype)

    tokens = convert_ids_to_tokens(inp, dm.tokenizer)
    fig, ax = plt.subplots(1, 1)
    for i in range(len(x)):
        ax.cla()
        ax.set_title(f"{tokens[i]}, label: {y[i].item()}")
        ax.plot(x[i, :, 0])
        plt.pause(0.01)
        input()

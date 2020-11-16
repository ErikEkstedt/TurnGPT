from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class TurnGPT(pl.LightningModule):
    def forward(self, input_ids, speaker_ids, **kwargs):
        """ labels are the the same as input_ids shift and padding fix inside model"""
        return self.model(input_ids, speaker_ids=speaker_ids, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        input_ids,
        speaker_ids,
        batch_size=-1,
        sp1_idx=None,
        sp2_idx=None,
        steps=50,
        top_k=5,
        temperature=1.0,
        sample=True,
        stop_at_turn_shift=True,
        use_pbar=False,
    ):
        """heavily basged on minGPT/utils in https://github.com/karpathy/minGPT """

        # collect samples and remove from processing when reaching a speaker token
        if stop_at_turn_shift:
            ks = []
            output_samples = []
            output_speaker = []

        # Get more samples by repeating the input by 'batch_size'
        # e.g. (1, 20) -> (10, 20) with batch_size=10
        # or   (5, 20) -> (50, 20) with batch_size=10
        if batch_size > 1:
            input_ids = torch.cat([input_ids] * batch_size)
            speaker_ids = torch.cat([speaker_ids] * batch_size)

        all_input_ids = input_ids.clone()  # store all data even if passt the block size
        all_speaker_ids = None
        if speaker_ids is not None:
            all_speaker_ids = (
                speaker_ids.clone()
            )  # store all data even if passt the block size
        if use_pbar:
            pbar = tqdm(range(steps), desc="TurnGPT Sampling")
        else:
            pbar = range(steps)

        input_ids = input_ids.to(self.device)
        speaker_ids = speaker_ids.to(self.device)

        # past = None
        for k in pbar:
            input_ids = input_ids[:, -self.model.block_size :]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[:, -self.model.block_size :]

            # TODO: use past in forward pass
            out = self(input_ids, speaker_ids=speaker_ids)
            lm_logits = (
                out["logits"][:, -1, :] / temperature
            )  # Prediction + scale w/ temp

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                lm_logits, idx = lm_logits.topk(k=top_k, dim=-1)
            probs = F.softmax(lm_logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                next_prob_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_idx = idx[
                    torch.arange(len(next_prob_idx)), next_prob_idx
                ].unsqueeze(-1)
                # next_idx = idx[:, next_prob_idx].unsqueeze(-1)
            else:
                next_idx = idx[:, :1]  # top choice

            if speaker_ids is not None:
                # fix speaker ids. Loop (only looking at last token so only looping over batch size -> fastish)
                next_speaker_ids = torch.zeros_like(next_idx)
                for i, (next_id, last_speaker) in enumerate(
                    zip(next_idx, speaker_ids[:, -1])
                ):
                    if next_id == sp1_idx or next_id == sp2_idx:
                        next_speaker = next_id
                    else:
                        next_speaker = last_speaker
                    next_speaker_ids[i] = last_speaker

            # append to the sequence and continue
            # past = transformer_outputs[1]
            input_ids = torch.cat((input_ids, next_idx), dim=-1)
            all_input_ids = torch.cat((all_input_ids, next_idx.cpu()), dim=-1)

            if speaker_ids is not None:
                speaker_ids = torch.cat((speaker_ids, next_speaker_ids), dim=-1)
                all_speaker_ids = torch.cat(
                    (all_speaker_ids, next_speaker_ids.cpu()), dim=-1
                )

            # print(next_idx)
            # print(all_input_ids.shape)
            if stop_at_turn_shift:
                sp1_batch, _ = torch.where(next_idx == sp1_idx)
                sp2_batch, _ = torch.where(next_idx == sp2_idx)
                sp_batch = torch.cat((sp1_batch, sp2_batch))
                if len(sp_batch) > 0:
                    # print('pre: ', all_input_ids.shape)
                    for i, b in enumerate(sp_batch):
                        ks.append(k + 1)
                        output_samples.append(all_input_ids[b])
                        output_speaker.append(all_speaker_ids[b])
                    keep_batches = [
                        ind
                        for ind in range(all_input_ids.shape[0])
                        if ind not in sp_batch
                    ]
                    input_ids = input_ids[keep_batches]
                    speaker_ids = speaker_ids[keep_batches]
                    all_input_ids = all_input_ids[keep_batches]
                    all_speaker_ids = all_speaker_ids[keep_batches]
                if len(all_input_ids) == 0:
                    return {
                        "output_ids": output_samples,
                        "output_speaker": output_speaker,
                        "k": ks,
                    }
        return {
            "output_ids": all_input_ids,
            "output_speaker": all_speaker_ids,
            "k": [k + 1] * all_input_ids.shape[0],
        }

    def loss_function(self, logits, labels):
        if self.pad_idx is not None:
            labels[labels == self.pad_idx] = -100  # don't train on these
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def training_step(self, batch, *args, **kwargs):
        input_ids, speaker_ids = batch[0], batch[1]
        # if self.hparams.general_turn_shift_token:
        #     inp_ids[inp_ids == self.sp1_idx] = self.ts_idx
        #     inp_ids[inp_ids == self.sp2_idx] = self.ts_idx

        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "avg_train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        """for multi-gpu"""
        return {"loss": batch_parts["loss"].mean()}

    def validation_step(self, batch, *args, **kwargs):
        input_ids, speaker_ids = batch[0], batch[1]
        # if self.hparams.general_turn_shift_token:
        #     inp_ids[inp_ids == self.sp1_idx] = self.ts_idx
        #     inp_ids[inp_ids == self.sp2_idx] = self.ts_idx

        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        speaker_ids = speaker_ids[:, :-1].contiguous()
        output = self(input_ids, speaker_ids)
        loss = self.loss_function(output["logits"], labels)
        self.log("val_loss", loss)
        return {"val_loss": loss}

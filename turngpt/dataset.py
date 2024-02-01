import os

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

from turngpt.tokenizer import SpokenDialogTokenizer

DATASETS = {"soda": "allenai/soda"}  # "faithdial": "McGill-NLP/FaithDial"}


def load_soda():
    """
    This is a separate function because changes makes the huggingface cashing be redone
    """
    tokenizer = SpokenDialogTokenizer()

    def process_soda(d):
        turns = [d["dialogue"][0]]
        spkr = d["speakers"][0]
        for i in range(1, len(d["dialogue"])):
            if spkr == d["speakers"][i]:
                turns[-1] += " " + d["dialogue"][i]
            else:
                turns.append(d["dialogue"][i])
                spkr = d["speakers"][i]
        tok = tokenizer(turns)
        return {
            "input_ids": tok["input_ids"],
            "speaker_ids": tok["speaker_ids"],
            "turns": turns,
            "dset": "soda",
        }

    dataset = load_dataset("allenai/soda")
    print("Preprocess soda")
    dset = dataset.map(process_soda, num_proc=os.cpu_count())
    dset = dset.remove_columns(
        [
            "head",
            "relation",
            "tail",
            "literal",
            "narrative",
            "PersonX",
            "PersonY",
            "PersonZ",
            "original_index",
            "split",
            "head_answer",
            "pmi_head_answer",
            "relation_tail_answer",
            "pmi_relation_tail_answer",
            "dialogue",
            "speakers",
        ]
    )
    dset.set_format(type="torch", columns=["input_ids", "speaker_ids", "turns", "dset"])
    return dset


class SodaDM(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = -1,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        if num_workers == -1:
            cpus = os.cpu_count()
            num_workers = 4 if cpus is None else cpus
        self.num_workers = num_workers
        self.pin_memory = True
        self.max_length = max_length
        self.tokenizer = SpokenDialogTokenizer()

    def prepare_data(self):
        _ = load_soda()

    def setup(self, stage: str):
        dset = load_soda()
        dset.set_format(
            type="torch", columns=["input_ids", "speaker_ids", "turns", "dset"]
        )
        if stage == "fit" or stage is None:
            self.train_dset = dset["train"]
            self.val_dset = dset["validation"]

        if stage == "test":
            self.test_dset = dset["test"]

    def collate_fn(self, batch):
        ret = self.tokenizer.pad(
            {"input_ids": [b["input_ids"][: self.max_length] for b in batch]}
        )
        ret["speaker_ids"] = self.tokenizer.pad(
            {"input_ids": [b["speaker_ids"][: self.max_length] for b in batch]}
        )["input_ids"]
        return ret

    def _dataset(self, dset, **kwargs):
        return DataLoader(
            dset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs
        )

    def train_dataloader(self):
        return self._dataset(self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self._dataset(self.val_dset)

    def test_dataloader(self):
        return self._dataset(self.test_dset)


if __name__ == "__main__":
    dm = SodaDM(batch_size=4)
    # dm.prepare_data()
    dm.setup("fit")

    import os

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    batch = next(iter(dm.train_dataloader()))

    batch["input_ids"].shape

    for batch in dm.train_dataloader():
        print(batch["input_ids"].shape)

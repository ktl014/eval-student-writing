"""Make data module for training and testing"""
import random
from typing import Optional, Sequence

import hydra
import os
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import ClassLabel, load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.common.constants import GenericConstants as gc
from src.common.utils import PROJECT_ROOT


def worker_init_fn(worker_id):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class MyDataModule(pl.LightningDataModule):
    """Build data module"""

    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        tokenizer: str,
        max_length: int,
    ):
        """Initialize data variables from .env file

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()

        Args:
            datasets (DictConfig): dataset contains train, val and test
            num_workers (DictConfig): number of thread/processes
               working parallel
            batch_size (DictConfig): batch size for train, val and test
            tokenizer (str): model's name
            max_length (int): max number of word per line (BERT can store
               max 512 words per line)
        """
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.labels: int = 0
        self.train_datasets: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        """Load dataset and split 80-train and 20-validation


        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.prepare_data()

        Returns:

        """
        # Split train to first 80%
        self.train_datasets = load_dataset(
            "csv",
            data_files={
                "train": self.datasets.train.path,
            },
            split="train[:80%]",
        )

        # Split val to last 20%
        self.val_datasets = load_dataset(
            "csv", data_files={"train": self.datasets.train.path},
            split="train[-20%:]"
        )

        # Save all unique labels
        dset_df = pd.read_csv(self.datasets.train.path)
        unique_labels = list(dset_df["discourse_type"].unique())
        self.labels = ClassLabel(names=unique_labels)

    def tokenize_and_label_encoding(self, example):
        """Tokenize features and encode label

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.tokenize_and_label_encoding(my_dataset.train_datasets)

        Args:
            example (datasets.Dataset): dataset

        Returns:
            tokens (transformers.BatchEncoding): tokenized and
                label-encoded dataset
        """
        tokens = self.tokenizer(
            example["discourse_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokens["discourse_type"] = self.labels.str2int(example[gc.LABEL])
        return tokens

    def setup(self, stage: Optional[str] = None):
        """Perform tokenizing, encoding and formatting dataset

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.setup()

        Args:
            stage (str): stage can either be fit, test, or None

        Returns:

        """
        # Here you should instantiate your datasets, you may also
        if stage == "fit" or stage is None:
            self.train_datasets = self.train_datasets.map(
                self.tokenize_and_label_encoding, batched=True
            )
            self.train_datasets.set_format(
                type="torch", columns=["input_ids", "attention_mask", gc.LABEL]
            )

            self.val_datasets = self.val_datasets.map(
                self.tokenize_and_label_encoding, batched=True
            )
            self.val_datasets.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", gc.LABEL],
                output_all_columns=True,
            )

        # if stage is None or stage == "test":
        #     self.test_datasets = [
        #         hydra.utils.instantiate(dataset_cfg)
        #         for dataset_cfg in self.datasets.test
        #     ]

    def train_dataloader(self) -> DataLoader:
        """Apply DataLoader on train dataset - shuffle, batch_size, and workers

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.train_dataloader()()

        Returns:
            Sequence[DataLoader]: return a Sequence of
            size = len(train_dataset) / batch_size
        """
        return DataLoader(
            self.train_datasets,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Apply DataLoader on val dataset - shuffle, batch_size, and workers

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.val_dataloader()

        Returns:
            Sequence[DataLoader]: return a Sequence of
            size = len(val_datasets) / batch_size
        """
        return DataLoader(
            self.val_datasets,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        """Apply DataLoader on val dataset - shuffle, batch_size, and workers

        Usage:
        >>> from src.pl_data.datamodule import MyDataModule
        >>> my_dataset = MyDataModule()
        >>> my_dataset.test_dataloader()

        Returns:
            Sequence[DataLoader]: return a Sequence of
            size = len(test_dataset) / batch_size
        """
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    """Execute script"""
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    datamodule.prepare_data()
    datamodule.setup()
    print(next(iter(datamodule.train_dataloader()))["input_ids"].shape)


if __name__ == "__main__":
    main()

import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.utils.warnings import warn_missing_pkg

from .dataset import Evaluation_Dataset, Train_Dataset, Semi_Dataset


class SPK_datamodule(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        trial_path=None,
        trial_paths=None,
        eval_warmup_epochs: int = 0,
        unlabel_csv_path = None,
        second: int = 2,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        pairs: bool = True,
        aug: bool = False,
        semi: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_csv_path = train_csv_path
        self.unlabel_csv_path = unlabel_csv_path
        self.second = second
        self.num_workers = num_workers
        self.batch_size = batch_size

        if trial_paths is None:
            if trial_path is None:
                raise ValueError("trial_path/trial_paths must be provided")
            trial_paths = [trial_path]
        self.trial_paths = list(trial_paths)
        self.eval_warmup_epochs = int(eval_warmup_epochs)

        self.pairs = pairs
        self.aug = aug
        print("second is {:.2f}".format(second))

    def train_dataloader(self) -> DataLoader:
        if self.unlabel_csv_path is None:
            train_dataset = Train_Dataset(self.train_csv_path, self.second, self.pairs, self.aug)
        else:
            train_dataset = Semi_Dataset(self.train_csv_path, self.unlabel_csv_path, self.second, self.pairs, self.aug)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def val_dataloader(self) -> DataLoader:
        eval_paths = []

        current_epoch = 0
        if getattr(self, "trainer", None) is not None:
            current_epoch = int(getattr(self.trainer, "current_epoch", 0))

        active_trial_paths = self.trial_paths
        if getattr(self, "trainer", None) is not None and getattr(self.trainer, "testing", False):
            active_trial_paths = self.trial_paths
        else:
            if self.eval_warmup_epochs > 0 and current_epoch < self.eval_warmup_epochs:
                active_trial_paths = self.trial_paths[:1]

        for p in active_trial_paths:
            trials = np.loadtxt(p, str)
            eval_paths.append(trials.T[1])
            eval_paths.append(trials.T[2])
            print("trials: {}".format(p))
            print("  number of enroll: {}".format(len(set(trials.T[1]))))
            print("  number of test: {}".format(len(set(trials.T[2]))))

        eval_path = np.unique(np.concatenate(tuple(eval_paths)))
        print("number of evaluation (union): {}".format(len(eval_path)))

        eval_dataset = Evaluation_Dataset(eval_path, second=-1)
        loader = torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=10,
            shuffle=False,
            batch_size=1,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()



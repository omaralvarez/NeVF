from ..utils import console
from .dataset_nevf_LES import NeVFDatasetLES

# 🐍 Python
import os

# 🍦 Vanilla PyTorch
import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler

# 👀 Torchvision
from torchvision import transforms

# 📊 Data
import pandas as pd
import numpy as np

# ⚡ PyTorch Lightning
import lightning.pytorch as pl

# 🏋️‍♀️ Weihts & Biases
import wandb


class LimitSampler(Sampler):
    def __init__(self, limit):
        self.limit = limit

    def __iter__(self):
        return iter(range(self.limit))

    def __len__(self):
        return self.limit


class DepthPadding(object):
    """Add padding to the depth dimension of 3D data in [D,H,W] format"""

    def __init__(self, padding):
        self.padding = padding

    def __call__(self, data):
        padded_shape = list(data.shape)
        padded_shape[0] += self.padding

        padded_data = np.zeros(padded_shape, dtype=data.dtype)

        # Set data at the beginning of the padded array
        padded_data[: data.shape[0], :, :] = data

        return padded_data


class NeVFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        sim_config,
        batch_size=128,
        num_workers=0,
        persistent=False,
        pin_memory=False,
        prefetch_factor=2,
        seed=42,
    ):
        super().__init__()
        self.config = config
        self.sim_config = sim_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent = persistent
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.seed = seed

        self.depth_padding = wandb.config.get("depth_padding", 0)

        # Create transform pipeline with padding
        if self.depth_padding > 0:
            self.transform = transforms.Compose(
                [DepthPadding(self.depth_padding), torch.from_numpy]
            )
        else:
            self.transform = transforms.Compose([torch.from_numpy])

        if wandb.config["model"] == "NeVFModel":
            self.dataset = NeVFDatasetLES
        else:
            raise NotImplementedError

    def prepare_data(self):
        self.data = pd.read_csv(os.path.join(self.sim_config["path"], "data.csv"))

    def setup(self, stage=None):
        self.length = len(self.data)
        inputs = self.data[: self.length]

        if stage == "fit" or stage is None:
            if not hasattr(self, "train") or self.train is None:
                console.print("📚 Training: {}".format(self.sim_config["path"]))
                self.train = self.dataset(
                    self.config, self.sim_config, inputs, self.transform, self.length
                )

            self.val = self.train

        if stage == "test" or stage is None:
            if not hasattr(self, "test") or self.test is None:
                self.test = (
                    hasattr(self, "train")
                    and self.train
                    or self.dataset(
                        self.config,
                        self.sim_config,
                        inputs,
                        self.transform,
                        self.length,
                    )
                )

        if stage == "predict" or stage is None:
            if not hasattr(self, "predict") or self.predict is None:
                self.predict = (
                    hasattr(self, "train")
                    and self.train
                    or self.dataset(
                        self.config,
                        self.sim_config,
                        inputs,
                        self.transform,
                        self.length,
                    )
                )

    def _loader(self, dataset, shuffle):
        if shuffle:
            # Randomly samples from the inflated dataset
            sampler = RandomSampler(dataset)
            batch_sampler = BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=True
            )
        else:
            # Sequentially samples exactly up to len frames
            sampler = LimitSampler(self.length)
            batch_sampler = BatchSampler(
                sampler, batch_size=self.batch_size, drop_last=False
            )

        kwargs = dict(
            dataset=dataset,
            batch_size=None,  # Retains advanced indexing
            sampler=batch_sampler,  # Batch-passing trick
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent if self.num_workers > 0 else False,
        )

        if self.num_workers > 0:
            if getattr(self, "prefetch_factor", None) is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            else:
                kwargs["prefetch_factor"] = 2

        return DataLoader(**kwargs)

    def train_dataloader(self):
        return self._loader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test, shuffle=False)

    def predict_dataloader(self):
        return self._loader(self.predict, shuffle=False)

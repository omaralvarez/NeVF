from ..utils import console, ProgressBar

# 🐍 Python
import os

# 🍦 Vanilla PyTorch
import torch
from torch.utils.data import Dataset

# 📊 Data
import numpy as np


class Dataset(Dataset):
    def __init__(self, config, path, data, transform, length):
        self.config = config
        self.path = path
        self.data = data
        self.transform = transform
        self.length = length
        self.multiplier = config["dataset_multiplier"]
        self.features_out = config["features_out"]
        self.depth = config["depth"]
        self.width = config["width"]
        self.height = config["height"]

        console.print("📟 Loading dataset into RAM...")
        self.fields_tensor_cache = []
        self.ts_cache = []

        with ProgressBar(transient=False).progress as p:
            for idx in p.track(range(self.length)):
                npz_path = os.path.join(self.path, self.data.loc[idx, "npz"])
                with np.load(npz_path) as npz:
                    keys = npz.files[-self.features_out :]
                    fields = [self.transform(npz[key]) for key in keys]

                fields_tensor = torch.stack(fields, 0)

                self.fields_tensor_cache.append(fields_tensor)
                self.ts_cache.append(
                    torch.tensor([self.__getinfo__(idx)], dtype=torch.float32)
                )

        console.print("📦 Stacking cache into contiguous tensors...")
        self.fields_tensor_cache = torch.stack(self.fields_tensor_cache)
        self.ts_cache = torch.stack(self.ts_cache)

        ts_min = self.ts_cache.min()
        ts_max = self.ts_cache.max()
        self.ts_cache = (self.ts_cache - ts_min) / (ts_max - ts_min)

    def __getinfo__(self, idx):
        return self.data.loc[idx, "ts"]

    def __len__(self):
        return self.length * self.multiplier

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # Convert list to a tensor and apply wrapping
            indices = torch.tensor(idx) % self.length
            return (
                self.ts_cache[indices],
                self.fields_tensor_cache[indices],
            )

        # Fallback for single-integer lookups (standard lightning sanity checks)
        real_idx = idx % self.length
        return (
            self.ts_cache[real_idx],
            self.fields_tensor_cache[real_idx],
        )

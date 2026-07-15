from ..utils import console, ProgressBar

# 🐍 Python
import os

# 🍦 Vanilla PyTorch
import torch
from torch.utils.data import Dataset

# 📊 Data
import numpy as np


class NeVFDatasetLES(Dataset):
    def __init__(self, config, sim_config, data, transform, length):
        self.config = config
        self.sim_config = config
        self.data = data
        self.transform = transform
        self.length = length
        self.multiplier = config["dataset_multiplier"]

        self.depth = config["depth"]
        self.width = config["width"]
        self.height = config["height"]
        self.path = sim_config["path"]
        self.start = sim_config["start_time"]
        self.end = sim_config["end_time"]

        console.print("📟 Loading dataset into RAM...")
        self.puvw_cache = []
        self.sf_cache = []
        self.ts_cache = []

        with ProgressBar(transient=False).progress as p:
            for idx in p.track(range(self.length)):
                npz_path = os.path.join(self.path, self.data.loc[idx, "npz"])
                with np.load(npz_path) as npz:
                    p, u, v, w = npz["p"], npz["u"], npz["v"], npz["w"]
                    sfx, sfy, sfz = npz["sfx"], npz["sfy"], npz["sfz"]

                puvw = torch.stack(
                    (
                        self.transform(p),
                        self.transform(u),
                        self.transform(v),
                        self.transform(w),
                    ),
                    0,
                )

                sf = torch.stack(
                    (
                        self.transform(sfx),
                        self.transform(sfy),
                        self.transform(sfz),
                    ),
                    0,
                )

                self.puvw_cache.append(puvw)
                self.sf_cache.append(sf)
                self.ts_cache.append(
                    torch.tensor([self.__getinfo__(idx)], dtype=torch.float32)
                )

        console.print("📦 Stacking cache into contiguous tensors...")
        self.puvw_cache = torch.stack(self.puvw_cache)
        self.sf_cache = torch.stack(self.sf_cache)
        self.ts_cache = torch.stack(self.ts_cache)

    def __getinfo__(self, idx):
        return (float(self.data.loc[idx, "ts"]) - self.start) / (self.end - self.start)

    def __len__(self):
        return self.length * self.multiplier

    def __getitem__(self, idx):
        if isinstance(idx, list):
            # Convert list to a tensor and apply wrapping
            indices = torch.tensor(idx) % self.length
            return (
                self.ts_cache[indices],
                self.puvw_cache[indices],
                self.sf_cache[indices],
            )

        # Fallback for single-integer lookups (standard lightning sanity checks)
        real_idx = idx % self.length
        return (
            self.ts_cache[real_idx],
            self.puvw_cache[real_idx],
            self.sf_cache[real_idx],
        )

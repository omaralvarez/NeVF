# 🍦 Vanilla PyTorch
import torch

# ⚡ PyTorch Lightning
import lightning.pytorch as pl

# 📊 Data
import numpy as np
import matplotlib.cm

# 🏋️‍♀️ Weights & Biases
import wandb


class PredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_ts, self.val_fields = val_samples
        self.val_ts = self.val_ts[:num_samples]
        self.val_fields = self.val_fields[:num_samples]

        self.depth = wandb.config["depth"]
        self.width = wandb.config["width"]
        self.height = wandb.config["height"]

    def apply_cmap(self, t):
        min = torch.min(t)
        max = torch.max(t)
        if min != max:
            t = (t - min) / (max - min)
        idx = torch.round(t * 255.0).type(torch.int64)
        map = matplotlib.colormaps.get_cmap("viridis")
        colors = torch.tensor(map.colors, dtype=torch.float32)

        return colors[idx[:, :, :], :].permute((0, 3, 1, 2))

    def on_validation_epoch_end(self, trainer, pl_module):
        val_ts = self.val_ts.to(device=pl_module.device)

        fields = self.val_fields.numpy()[:, :, 0, :, :]
        fields_vid = (fields * 255.0).astype(np.uint8)[:, 0:3, :, :]

        # .numpy() doesn't officially support bfloat16, cast to float32 first
        preds = pl_module(None, val_ts).float().cpu().numpy()[:, :, 0, :, :]
        preds_video = (preds * 255.0).astype(np.uint8)[:, 0:3, :, :]

        trainer.logger.experiment.log(
            {
                "flow": [
                    vid
                    for vid in (
                        wandb.Video(
                            preds_video, caption="Predicted fields", fps=4, format="gif"
                        ),
                        wandb.Video(
                            fields_vid, caption="Real fields", fps=4, format="gif"
                        ),
                    )
                ],
                "global_step": trainer.global_step,
            }
        )

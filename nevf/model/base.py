# 🐍 Python
from pathlib import Path
import os

# 🍦 Vanilla PyTorch
import torch

# ⚡ PyTorch Lightning
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio
import lightning.pytorch as pl

# 🏋️‍♀️ Weights & Biases
import wandb


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.single_decoder = wandb.config["single_decoder"]

        self.mape = MeanAbsolutePercentageError()

        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0))

        # Build spatial coordinates (xyz_grid) to avoid computing on CPU/Workers
        x = torch.linspace(0, 1, wandb.config["width"] + 1)[:-1]
        y = torch.linspace(0, 1, wandb.config["height"] + 1)[:-1]
        z = torch.linspace(0, 1, wandb.config["depth"] + 1)[:-1]

        xyz = torch.stack(torch.meshgrid(x, y, z, indexing="xy"), dim=-1)
        xyz = xyz.permute(3, 2, 0, 1).contiguous().float()  # [3, D, H, W]
        self.register_buffer("xyz_grid", xyz, persistent=False)

        if self.single_decoder:
            self.mse = MeanSquaredError()
        else:
            self.automatic_optimization = False

            self.mse = MeanSquaredError()

    def setup(self, stage=None):
        pass

    def forward(self, coords, ts):
        b = ts.shape[0]

        if coords is None:
            coords = self.xyz_grid

        if coords.dim() == 4:
            coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)

        if self.single_decoder:
            return self.model(coords, ts).contiguous()
        else:
            return torch.cat(
                [
                    self.model_p(coords, ts),
                    self.model_u(coords, ts),
                    self.model_v(coords, ts),
                    self.model_w(coords, ts),
                ],
                dim=1,
            )

    def shared_step(self, batch, stage):
        if len(batch) == 2:
            ts, fields = batch
        else:
            _, ts, fields = batch

        if not self.single_decoder:
            p_opt, u_opt, v_opt, w_opt = self.optimizers()

        pred = self.forward(self.xyz_grid, ts)

        mse = self.mse(pred, fields)

        if not self.single_decoder and stage == "train":
            loss_p = self.mse_p(
                pred[:, 0 : wandb.config["depth"], :, :].contiguous(),
                fields[:, 0 : wandb.config["depth"], :, :].contiguous(),
            )
            p_opt.zero_grad()
            self.manual_backward(loss_p, retain_graph=True)
            p_opt.step()

            loss_u = self.mse_u(
                pred[
                    :, wandb.config["depth"] : wandb.config["depth"] * 2, :, :
                ].contiguous(),
                fields[
                    :, wandb.config["depth"] : wandb.config["depth"] * 2, :, :
                ].contiguous(),
            )
            u_opt.zero_grad()
            self.manual_backward(loss_u, retain_graph=True)
            u_opt.step()

            loss_v = self.mse_v(
                pred[
                    :, wandb.config["depth"] * 2 : wandb.config["depth"] * 3, :, :
                ].contiguous(),
                fields[
                    :, wandb.config["depth"] * 2 : wandb.config["depth"] * 3, :, :
                ].contiguous(),
            )
            v_opt.zero_grad()
            self.manual_backward(loss_v, retain_graph=True)
            v_opt.step()

            loss_w = self.mse_w(
                pred[
                    :, wandb.config["depth"] * 3 : wandb.config["depth"] * 4, :, :
                ].contiguous(),
                fields[
                    :, wandb.config["depth"] * 3 : wandb.config["depth"] * 4, :, :
                ].contiguous(),
            )
            w_opt.zero_grad()
            self.manual_backward(loss_w)
            w_opt.step()

        # MAPE
        mape = self.mape(
            pred[:, :, : wandb.config["depth"], :, :],
            fields[:, :, : wandb.config["depth"], :, :],
        )

        # PSNR
        psnr = self.psnr(
            pred[:, :, : wandb.config["depth"], :, :],
            fields[:, :, : wandb.config["depth"], :, :],
        )

        loss = mse

        # logging metrics we calculated
        self.log(stage + "/MSE", mse, on_step=False, on_epoch=True)
        self.log(stage + "/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(stage + "/MAPE", mape, on_step=False, on_epoch=True)
        self.log(stage + "/PSNR", psnr, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        if wandb.config["upload"]:
            # validation_step_outputs = self.step_outputs
            model_filename = f"WindNet_{str(self.global_step).zfill(5)}.onnx"
            model_path = os.path.join("models", model_filename)
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.onnx.export(self, self.dummy_input, model_path, opset_version=17)
            artifact = wandb.Artifact(name="WindNet.ckpt", type="model")
            artifact.add_file(model_path)
            self.logger.experiment.log_artifact(artifact)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        if wandb.config["upload"]:
            model_filename = "WindNet_Final.onnx"
            model_path = os.path.join("models", model_filename)
            self.to_onnx(model_path, self.dummy_input, export_params=True)
            artifact = wandb.Artifact(name="WindNet.ckpt", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    def predict_step(self, batch, batch_idx):
        if len(batch) == 3:
            ts, _, _ = batch
        else:
            _, ts, _, _ = batch

        pred = self.forward(self.xyz_grid, ts).contiguous()

        return pred

    def configure_optimizers(self):
        if self.single_decoder:
            return torch.optim.Adam(self.parameters(), lr=wandb.config["lr"])
        else:
            p_opt = torch.optim.Adam(self.model_p.parameters(), lr=wandb.config["lr"])
            u_opt = torch.optim.Adam(self.model_u.parameters(), lr=wandb.config["lr"])
            v_opt = torch.optim.Adam(self.model_v.parameters(), lr=wandb.config["lr"])
            w_opt = torch.optim.Adam(self.model_w.parameters(), lr=wandb.config["lr"])

            return p_opt, u_opt, v_opt, w_opt

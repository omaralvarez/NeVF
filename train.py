from nevf.utils import console, Config
from nevf.model.datamodule import NeVFDataModule
from nevf.model.loggers import PredictionLogger
import nevf.model as model

# 🐍 Python
import argparse
import datetime
import os
import warnings

# 🍦 Vanilla PyTorch
import torch

# ⚡ PyTorch Lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping, ModelCheckpoint

# 🏋️‍♀️ Weights & Biases
import wandb

# ⚡ 🤝 🏋️‍♀️
from lightning.pytorch.loggers import WandbLogger

os.environ["WANDB_QUIET"] = "true"
os.environ["POSSIBLE_USER_WARNINGS"] = "off"

parser = argparse.ArgumentParser("NeVF", description="Script to train our models.")
parser.add_argument(
    "-c", "--config", help="Path to JSON config file.", type=str, required=True
)
parser.add_argument("-s", "--seed", type=int, help="Set random seed.")
parser.add_argument(
    "-C", "--compile", action="store_true", help="Compile model sub-modules."
)


def main(args):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(datetime.now().timestamp())

    console.print("🎲 Random seed: {}".format(seed))

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    config = Config(args.config)

    wandb_logger = WandbLogger(project=config["project"], config=config.to_dict())

    # Setup data
    cwdm = NeVFDataModule(
        config,
        config["simulation"],
        batch_size=wandb_logger.experiment.config["batch_size"],
        num_workers=wandb_logger.experiment.config["num_workers"],
        persistent=wandb_logger.experiment.config["persistent"],
        pin_memory=wandb_logger.experiment.config["pin_memory"],
        prefetch_factor=wandb_logger.experiment.config["prefetch_factor"],
    )
    cwdm.prepare_data()
    cwdm.setup()

    # Grab samples to log predictions on
    val_samples = next(iter(cwdm.val_dataloader()))
    test_samples = next(iter(cwdm.test_dataloader()))

    trainer = pl.Trainer(
        num_nodes=wandb_logger.experiment.config["num_nodes"],
        devices=wandb_logger.experiment.config["devices"],
        accelerator=wandb_logger.experiment.config["accelerator"],
        precision=wandb_logger.experiment.config["precision"],
        logger=wandb_logger,  # W&B integration
        log_every_n_steps=wandb_logger.experiment.config["log_every_n_steps"],
        max_epochs=wandb_logger.experiment.config["epochs"],  # Number of epochs
        deterministic=wandb_logger.experiment.config[
            "deterministic"
        ],  # Keep it deterministic
        benchmark=wandb_logger.experiment.config[
            "benchmark"
        ],  # Use cudnn auto-tuner to find the best algorithm to use for your hardware
        check_val_every_n_epoch=wandb_logger.experiment.config[
            "check_val_every_n_epoch"
        ],  # Check validation every epoch
        callbacks=[
            PredictionLogger(
                val_samples, num_samples=wandb_logger.experiment.config["batch_size"]
            ),
            RichProgressBar(),
            ModelCheckpoint(
                save_top_k=1,  # Save only the best ckpt for monitoring metric
                monitor=wandb_logger.experiment.config["monitor"],
                mode=wandb_logger.experiment.config["mode"],  # Loss needs to be min
            ),
            EarlyStopping(
                monitor=wandb_logger.experiment.config["monitor"],
                mode=wandb_logger.experiment.config["mode"],
                patience=wandb_logger.experiment.config["patience"],
            ),
        ],
    )

    # Setup model
    mdl = getattr(model, wandb_logger.experiment.config["model"])()

    # Compile the inner PyTorch nn.Modules with TorchDynamo
    if args.compile:
        console.print("⚡ Compiling model with torch.compile...")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 128
        compile_kwargs = {
            "mode": "reduce-overhead",
            "fullgraph": False,
            "dynamic": True,
        }
        if hasattr(mdl, "model"):
            mdl.model = torch.compile(mdl.model, **compile_kwargs)
        if hasattr(mdl, "model_p"):
            mdl.model_p = torch.compile(mdl.model_p, **compile_kwargs)
            mdl.model_u = torch.compile(mdl.model_u, **compile_kwargs)
            mdl.model_v = torch.compile(mdl.model_v, **compile_kwargs)
            mdl.model_w = torch.compile(mdl.model_w, **compile_kwargs)

    # Fit the model
    trainer.fit(mdl, cwdm)

    # Evaluate the model on a test set using best model
    trainer.test(datamodule=cwdm, ckpt_path="best")

    if args.compile:
        state_dict = mdl.state_dict()

        # Remove the '_orig_mod.' prefix added by torch.compile
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        clean_state_dict = mdl.state_dict()

    mdl = getattr(model, config["model"])()
    mdl.load_state_dict(clean_state_dict)
    mdl.eval()

    # Save final model
    torch.save(mdl.state_dict(), "model.pth")

    wandb.finish()


if __name__ == "__main__":
    main(parser.parse_args())

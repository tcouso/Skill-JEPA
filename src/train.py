import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import ModelConfig
from src.datamodule import PlatonicDataModule
from src.system import ModelSystem
from src.callbacks import WandbReconstructionCallback


def main(args):
    config = ModelConfig.from_yaml(args.config)
    pl.seed_everything(config.seed, workers=True)

    run_name = f"ActSiamMAE_mask{config.start_masking_ratio}_to_{config.target_masking_ratio}_bs{config.batch_size}_lr{config.base_learning_rate}"

    wandb_kwargs = {"project": "ActiveSiamMAE", "name": run_name, "log_model": "all"}

    if args.resume_id:
        wandb_kwargs["id"] = args.resume_id
        wandb_kwargs["resume"] = "must"

    wandb_logger = WandbLogger(**wandb_kwargs)
    wandb_logger.experiment.config.update(vars(config))

    data_module = PlatonicDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"./checkpoints/{wandb_logger.version}",
        filename="sample-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    reconstruction_callback = WandbReconstructionCallback(config)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy if config.devices != 1 else "auto",
        callbacks=[checkpoint_callback, reconstruction_callback],
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
    )
    system = ModelSystem(config)
    trainer.fit(system, datamodule=data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--resume_id", type=str, default=None, help="WandB run ID to resume"
    )
    main(parser.parse_args())

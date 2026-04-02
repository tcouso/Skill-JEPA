import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import ModelConfig
from src.datamodule import LeWMDataModule
from src.system import ModelSystem

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: ModelConfig):
    # If config is a DictConfig (from Hydra), we might want to convert to our dataclass
    # for full type safety, though Hydra's structured configs can do this automatically.

    pl.seed_everything(config.seed, workers=True)

    run_name = f"LeWM_{config.predictor.mode}_bs{config.training.batch_size}_lr{config.training.base_learning_rate}"
    wandb_kwargs = {"project": "SkillJEPA", "name": run_name, "log_model": "all"}

    # Hydra allows us to pass these via command line more easily, e.g., ++wandb.id=...
    # For now, let's keep the core logic simple.

    wandb_logger = WandbLogger(**wandb_kwargs)
    wandb_logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    data_module = LeWMDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"./checkpoints/{wandb_logger.version}",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy if config.devices != 1 else "auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.training.log_every_n_steps,
        logger=wandb_logger,
    )

    system = ModelSystem(config)
    trainer.fit(system, datamodule=data_module)

if __name__ == "__main__":
    main()
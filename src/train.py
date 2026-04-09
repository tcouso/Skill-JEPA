import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.config import JEPAConfig
from src.datamodule import LeWMDataModule
from src.system import ModelSystem

cs = ConfigStore.instance()
cs.store(name="base_config", node=JEPAConfig)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    config: JEPAConfig = OmegaConf.to_object(cfg)

    pl.seed_everything(config.seed, workers=True)

    run_name = f"LeWM_{config.predictor.mode}_bs{config.training.batch_size}_lr{config.training.base_learning_rate}"
    wandb_kwargs = {"project": "SkillJEPA", "name": run_name, "log_model": "all"}
    wandb_logger = WandbLogger(**wandb_kwargs)
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    data_module = LeWMDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=f"./checkpoints/{wandb_logger.version}",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if config.devices != 1 else "auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.training.log_every_n_steps,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        logger=wandb_logger,
    )

    system = ModelSystem(config)
    trainer.fit(system, datamodule=data_module)


if __name__ == "__main__":
    main()

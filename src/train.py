import os
import sys

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.config import JEPAConfig, reject_stray_experiment, wandb_enabled
from src.training import DataModule, MpcEvalCallback, SkillJEPASystem


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    reject_stray_experiment(cfg)
    config: JEPAConfig = OmegaConf.to_object(cfg)

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config.seed, workers=True)

    checkpoint_base = config.checkpoint_dir
    if not os.path.isabs(checkpoint_base):
        checkpoint_base = os.path.join(get_original_cwd(), checkpoint_base)
    run_id = config.run_id or wandb.util.generate_id()
    run_dir = os.path.join(checkpoint_base, run_id)

    encoder_tag = str(cfg.get("action_encoder", "") or "")

    experiment_name = HydraConfig.get().runtime.choices.get("experiment", None)
    if experiment_name:
        run_name = experiment_name
        if encoder_tag == "vae":
            run_name += f"-kld{config.action.kld_weight}"
    else:
        model_tag = "SkillJEPA" if config.action.representation == "latent" else "RawJEPA"
        env_tag = config.world.env_name.split("/")[-1].split("-")[0].lower()
        run_name = f"{model_tag}-{env_tag}-{encoder_tag}-bs{config.training.batch_size}"
    run_name += f"-seed{config.seed}"
    if wandb_enabled():
        logger = WandbLogger(
            project="SkillJEPA",
            name=run_name,
            id=run_id,
            resume="allow",
            log_model=False,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        logger = CSVLogger(save_dir=run_dir, name="", version="")
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    print(f"[train] run {run_name}, id {run_id}, checkpoints in {run_dir}")

    data_module = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=run_dir,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
    )

    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]
    if config.evaluation.enabled:
        callbacks.append(MpcEvalCallback(
            cfg=cfg,
            frequency=config.evaluation.frequency,
            render_frequency=config.evaluation.render_frequency,
        ))

    # Lightning crashes the DDP ranks if log_every_n_steps exceeds a fractional batch limit.
    limit = config.training.limit_train_batches
    log_every_n_steps = (
        1 if isinstance(limit, float) and limit < 1.0
        else config.training.log_every_n_steps
    )

    _devices = config.devices
    _use_ddp = (isinstance(_devices, int) and _devices > 1) or (
        _devices == "auto" and torch.cuda.device_count() > 1
    )
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.accelerator,
        devices=_devices,
        precision=config.precision,
        strategy=(
            DDPStrategy(find_unused_parameters=True) if _use_ddp else "auto"
        ),
        sync_batchnorm=True,
        enable_progress_bar=sys.stdout.isatty(),
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        gradient_clip_val=1.0,
        logger=logger,
    )

    system = SkillJEPASystem(config)

    param_counts = {
        "params/total": sum(p.numel() for p in system.parameters()),
        "params/obs_encoder": sum(p.numel() for p in system.obs_enc.parameters()),
        "params/obs_proj": sum(p.numel() for p in system.obs_proj.parameters()),
        "params/pred_proj": sum(p.numel() for p in system.pred_proj.parameters()),
        "params/predictor": sum(p.numel() for p in system.predictor.parameters()),
    }
    if system.skill_encoder is not None:
        param_counts["params/skill_encoder"] = sum(
            p.numel() for p in system.skill_encoder.parameters()
        )
    logger.log_hyperparams(param_counts)

    last_ckpt = os.path.join(run_dir, "last.ckpt")
    ckpt_path = config.ckpt_path or (last_ckpt if os.path.exists(last_ckpt) else None)
    trainer.fit(system, datamodule=data_module, ckpt_path=ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    main()

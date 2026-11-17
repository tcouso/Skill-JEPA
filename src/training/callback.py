from pathlib import Path

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src.evaluation.mpc_eval import run_mpc_eval


class MpcEvalCallback(pl.Callback):
    """MPC eval every `frequency` epochs on rank 0, with contact sheets every `render_frequency`."""

    def __init__(self, cfg: DictConfig, frequency: int, render_frequency: int = 0):
        self.cfg = cfg
        self.frequency = frequency
        self.render_frequency = render_frequency

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:
            return

        epoch = trainer.current_epoch
        is_last = epoch == trainer.max_epochs - 1
        if (epoch + 1) % self.frequency != 0 and not is_last:
            return

        render = self.render_frequency > 0 and (
            (epoch + 1) % self.render_frequency == 0 or is_last
        )

        pl_module.jepa.eval()
        metrics, _, contact_sheets = run_mpc_eval(
            pl_module.jepa, pl_module.config, self.cfg, render=render
        )
        pl_module.jepa.train()

        trainer.logger.log_metrics(
            {f"eval/{k}": v for k, v in metrics.items()}, step=trainer.global_step
        )
        if not contact_sheets:
            return
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {f"render/contact_sheet_{i}": wandb.Image(sheet)
                 for i, sheet in enumerate(contact_sheets)}
            )
            return
        sheet_dir = Path(trainer.log_dir) / "contact_sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for i, sheet in enumerate(contact_sheets):
            sheet.save(sheet_dir / f"epoch{epoch:03d}_{i:02d}.png")

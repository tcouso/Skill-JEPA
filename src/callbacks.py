import torch
import wandb
import torchvision
from pytorch_lightning.callbacks import Callback

from src.config import ActSiamMAEConfig


class WandbReconstructionCallback(Callback):
    def __init__(self, config: ActSiamMAEConfig):
        super().__init__()
        self.config = config
        self.log_every_n_steps = config.log_every_n_steps
        self.num_samples = config.recon_num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _ = outputs
        _ = batch_idx
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            self._log_reconstruction(trainer, pl_module, batch)

    def _log_reconstruction(self, trainer, pl_module, batch):
        pl_module.eval()

        with torch.no_grad():
            past, future, masked, reconstructed = pl_module.reconstruct(batch)

        pl_module.train()

        past = past[: self.num_samples].clamp(0, 1)
        future = future[: self.num_samples].clamp(0, 1)
        masked = masked[: self.num_samples].clamp(0, 1)
        reconstructed = reconstructed[: self.num_samples].clamp(0, 1)

        grid_rows = []
        for i in range(self.num_samples):
            row = torch.stack([past[i], future[i], masked[i], reconstructed[i]])
            grid_rows.append(torchvision.utils.make_grid(row, nrow=4, padding=2))

        final_grid = torchvision.utils.make_grid(
            torch.stack(grid_rows), nrow=1, padding=4
        )

        trainer.logger.experiment.log(
            {
                "reconstructions": wandb.Image(
                    final_grid,
                    caption="Columns: Past(t) | Future(t+1) | Masked | Reconstructed",
                ),
                "global_step": trainer.global_step,
            }
        )

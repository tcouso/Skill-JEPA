import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Tuple
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.model import (
    ActSiamMAEConfig,
    ActSiamMAEEncoder,
    ActSiamMAEDecoder,
    ActSiamMAEDepatchifier,
)


class ActSiamMAESystem(pl.LightningModule):
    def __init__(self, config: ActSiamMAEConfig):
        super().__init__()
        self.save_hyperparameters("config")
        
        self.config = config
        
        self.num_channels = config.num_channels
        self.frame_size = config.frame_size
        self.batch_size = config.batch_size
        self.base_learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.betas = config.betas
        self.max_epochs = config.max_epochs
        self.warmup_epochs = config.warmup_epochs

        self.encoder = ActSiamMAEEncoder(config)
        self.decoder = ActSiamMAEDecoder(config)
        self.depatchifier = ActSiamMAEDepatchifier(config)

    def _shared_step(self, batch) -> torch.Tensor:
        past_frames = (
            batch["images"][:, :-1, :, :, :]
            .reshape(-1, self.num_channels, self.frame_size, self.frame_size)
            .float()
        )
        future_frames = (
            batch["images"][:, 1:, :, :, :]
            .reshape(-1, self.num_channels, self.frame_size, self.frame_size)
            .float()
        )

        past_embeddings, future_embeddings, mask, ids_restore = self.encoder(
            past_frames, future_frames
        )
        future_patches = self.encoder.patch_layer(future_frames)
        pred_patches = self.decoder(past_embeddings, future_embeddings, ids_restore)

        loss = F.mse_loss(pred_patches[mask.bool()], future_patches[mask.bool()])
        return loss

    def training_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            batch_size=self.config.batch_size,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log(
            "val_loss",
            loss,
            batch_size=self.config.batch_size,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def reconstruct(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        past_frames = (
            batch["images"][:, :-1, :, :, :]
            .reshape(-1, self.num_channels, self.frame_size, self.frame_size)
            .float()
        )
        future_frames = (
            batch["images"][:, 1:, :, :, :]
            .reshape(-1, self.num_channels, self.frame_size, self.frame_size)
            .float()
        )

        past_embeddings, future_embeddings, mask, ids_restore = self.encoder(
            past_frames, future_frames
        )
        pred_patches = self.decoder(past_embeddings, future_embeddings, ids_restore)

        future_patches = self.encoder.patch_layer(future_frames)

        masked_patches = future_patches.clone()
        masked_patches[mask.bool()] = 0.0

        reconstructed_frames = self.depatchifier(pred_patches)
        masked_frames = self.depatchifier(masked_patches)

        return past_frames, future_frames, masked_frames, reconstructed_frames


    # TODO: Understand these changes, and how do they relate with
    # representation collapse observed during the last training session
    def configure_optimizers(self) -> OptimizerLRScheduler:
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        absolute_lr = self.base_learning_rate * (self.batch_size / 256.0)

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=absolute_lr,
            betas=self.betas,
        )

        decay_epochs = self.max_epochs - self.warmup_epochs

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=self.warmup_epochs
        )
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_epochs, eta_min=0.0
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

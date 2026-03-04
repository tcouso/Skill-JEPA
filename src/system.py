import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from src.model import ActSiamMAEConfig, ActSiamMAEEncoder, ActSiamMAEDecoder, ActSiamMAEDepatchifier

class ActSiamMAESystem(pl.LightningModule):
    def __init__(self, config: ActSiamMAEConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_channels = config.num_channels
        self.frame_size = config.frame_size
        self.encoder = ActSiamMAEEncoder(config)
        self.decoder = ActSiamMAEDecoder(config)
        self.depatchifier = ActSiamMAEDepatchifier(config)

    def _shared_step(self, batch) -> torch.Tensor:
        past_frames = batch['images'][:, :-1, :, :, :].reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        future_frames = batch['images'][:, 1:, :, :, :].reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        
        past_embeddings, future_embeddings, mask, ids_restore = self.encoder(past_frames, future_frames)
        future_patches = self.encoder.patch_layer(future_frames)
        pred_patches = self.decoder(past_embeddings, future_embeddings, ids_restore)
        
        loss = F.mse_loss(pred_patches[mask.bool()], future_patches[mask.bool()])
        return loss

    def training_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, batch_size=self.config.batch_size, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("val_loss", loss, batch_size=self.config.batch_size, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def reconstruct(self, batch):
        
        past_frames = batch['images'][:, :-1, :, :, :].reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        future_frames = batch['images'][:, 1:, :, :, :].reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()

        past_embeddings, future_embeddings, mask, ids_restore = self.encoder(past_frames, future_frames)
        pred_patches = self.decoder(past_embeddings, future_embeddings, ids_restore)
        
        future_patches = self.encoder.patch_layer(future_frames)
        
        masked_patches = future_patches.clone()
        masked_patches[mask.bool()] = 0.0
        
        reconstructed_frames = self.depatchifier(pred_patches)
        masked_frames = self.depatchifier(masked_patches)
        
        return past_frames, future_frames, masked_frames, reconstructed_frames

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

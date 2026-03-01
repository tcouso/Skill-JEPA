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
        past_frames = batch['images'][:, :-1, :, :, :].permute(0, 1, 4, 2, 3).reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        future_frames = batch['images'][:, 1:, :, :, :].permute(0, 1, 4, 2, 3).reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        
        past_embeddings, future_embeddings, mask, ids_restore = self.encoder(past_frames, future_frames)
        future_patches = self.encoder.patch_layer(future_frames)
        pred_patches = self.decoder(past_embeddings, future_embeddings, ids_restore)
        
        loss = F.mse_loss(pred_patches[mask.bool()], future_patches[mask.bool()])
        return loss

    def training_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def reconstruct(self, batch):
        past_frames = batch['images'][:, :-1, :, :, :].permute(0, 1, 4, 2, 3).reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()
        future_frames = batch['images'][:, 1:, :, :, :].permute(0, 1, 4, 2, 3).reshape(-1, self.num_channels, self.frame_size, self.frame_size).float()

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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.dataset import read_platonic_solids_dataset
    
    print("--- Initializing ActiveSiamMAE System with Real Data ---")
    config = ActSiamMAEConfig()
    system = ActSiamMAESystem(config)
    
    print(f"Loading data from: 'data/medium_data/'")
    dataset = read_platonic_solids_dataset("data/medium_data/")
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    batch = next(iter(train_loader))
    images = batch['images']
    print(f"\nReal Batch Shape: {images.shape}") # Expected: (Batch, Seq_Len, Channels, Height, Width)

    velocities = batch['actions']
    print(f"Velocities Shape: {velocities.shape}") # Expected: (Batch, Seq_Len, 3)

    print("\n--- Checking Sequence Slicing ---")
    
    seq_len = images.shape[1]
    print(f"Sequence Length in Data: {seq_len}")
    
    if seq_len > 1:
        past_frames = images[:, :-1, ...].permute(0, 1, 4, 2, 3)
        past_frames = past_frames.reshape(-1, config.num_channels, config.frame_size, config.frame_size).float()
        
        future_frames = images[:, 1:, ...].permute(0, 1, 4, 2, 3)
        future_frames = future_frames.reshape(-1, config.num_channels, config.frame_size, config.frame_size).float()
        flatten_velocities = velocities.reshape(-1, 3)

        
        print(f"Past Frames (Flattened):   {past_frames.shape}")
        print(f"Future Frames (Flattened):   {future_frames.shape}")
        print(f"Velocities (Flattened):   {flatten_velocities.shape}")
        
        print("\n--- Running System.training_step() ---")
        loss = system.training_step(batch)
        print(f"Loss successfully calculated: {loss.item()}")
        
    else:
        print("Error: Sequence length is 1. Cannot perform temporal slicing (t -> t+1).")
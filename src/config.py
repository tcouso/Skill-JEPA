import torch
import yaml
from dataclasses import dataclass, fields


@dataclass
class ActSiamMAEConfig:
    # General parameters
    seed: int = 42
    data_dir: str = "data/medium_data/"
    train_ratio: float = 0.8
    num_workers: int = 4

    # Model Hyperparameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_channels: int = 3
    patch_size: int = 16
    grid_side_length: int = 14
    hidden_dim: int = 512
    frame_size: int = 224
    num_attn_heads: int = 8
    seq_length: int = grid_side_length * grid_side_length
    head_dimension: int = hidden_dim // num_attn_heads
    masking_ratio: float = 0.75
    encoder_num_layers: int = 4
    decoder_num_layers: int = 4

    # Training Hyperparameters
    batch_size: int = 64
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    max_epochs: int = 800
    warmup_epochs: int = 40

    # Lightning Trainer Params
    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    log_every_n_steps: int = 10
    recon_log_every_n_steps: int = 100

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Only pass keys that actually exist in the dataclass
        valid_keys = {f.name for f in fields(cls)}
        config_dict = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**config_dict)

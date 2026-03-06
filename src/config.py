import torch
import yaml
from dataclasses import dataclass, fields

@dataclass
class ActSiamMAEConfig:
    seed: int = 42
    train_urls: str = "data/wds_sample_trajectories/train/platonic-{0000..0003}.tar"
    val_urls: str = "data/wds_sample_trajectories/val/platonic-0000.tar"
    num_workers: int = 4

    wds_shard_shuffle_size: int = 100
    wds_sample_shuffle_size: int = 1000

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_channels: int = 3
    patch_size: int = 8
    grid_side_length: int = 28
    hidden_dim: int = 512
    frame_size: int = 224
    num_attn_heads: int = 8
    seq_length: int = grid_side_length * grid_side_length
    head_dimension: int = hidden_dim // num_attn_heads
    encoder_num_layers: int = 4
    decoder_num_layers: int = 4

    batch_size: int = 40
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    start_masking_ratio: float = 0.75
    target_masking_ratio: float = 0.90
    max_epochs: int = 400
    warmup_epochs: int = 40
    masking_schedule_epochs: int = 100

    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    log_every_n_steps: int = 10
    recon_log_every_n_steps: int = 100
    recon_num_samples: int = 4

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        valid_keys = {f.name for f in fields(cls)}
        config_dict = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**config_dict)
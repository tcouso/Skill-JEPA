import torch
import yaml
from dataclasses import dataclass, fields, field
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    # --- Identification & Metadata ---
    vit_variant: Optional[str] = "tiny" 
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # --- Data (WebDataset) ---
    train_urls: str = "data/wds_sample_trajectories/train/platonic-{0000..0003}.tar"
    val_urls: str = "data/wds_sample_trajectories/val/platonic-0000.tar"
    wds_shard_shuffle_size: int = 100
    wds_sample_shuffle_size: int = 1000

    # --- Observation Encoder (ViT Standards) ---
    frame_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    obs_encoder_hidden_dim: int = 192
    obs_encoder_num_attn_heads: int = 3
    obs_encoder_num_layers: int = 12
    obs_encoder_mlp_ratio: int = 4
    grid_side_length: int = field(init=False)
    obs_encoder_seq_length: int = field(init=False)
    obs_encoder_head_dimension: int = field(init=False)
    obs_encoder_mlp_dim: int = field(init=False)

    # --- Action Encoder ---
    action_space_dim: int = 3
    action_sequence_length: int = 64
    action_encoder_num_layers: int = 2
    action_encoder_hidden_dim: int = 64
    action_encoder_num_attn_heads: int = 8
    action_encoder_mlp_ratio: int = 4
    action_encoder_head_dimension: int = field(init=False)

    # --- Training & Masking ---
    batch_size: int = 40
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    start_masking_ratio: float = 0.75
    target_masking_ratio: float = 0.90
    max_epochs: int = 400
    warmup_epochs: int = 40
    masking_schedule_epochs: int = 100

    # --- Infrastructure ---
    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    log_every_n_steps: int = 10
    recon_log_every_n_steps: int = 100
    recon_num_samples: int = 4

    def __post_init__(self):
        """Standardizes dimensions and prevents architectural drift."""
        self.grid_side_length = self.frame_size // self.patch_size
        self.obs_encoder_seq_length = self.grid_side_length ** 2
        self.obs_encoder_head_dimension = self.obs_encoder_hidden_dim // self.obs_encoder_num_attn_heads
        self.obs_encoder_mlp_dim = int(self.obs_encoder_hidden_dim * self.obs_encoder_mlp_ratio)
        self.action_encoder_head_dimension = self.action_encoder_hidden_dim // self.action_encoder_num_attn_heads

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)

        final_params = {}

        variant = yaml_data.get("vit_variant")
        if variant:
            standards = {
                "tiny": {"obs_encoder_hidden_dim": 192, "obs_encoder_num_layers": 12, "obs_encoder_num_attn_heads": 3},
                "small": {"obs_encoder_hidden_dim": 384, "obs_encoder_num_layers": 12, "obs_encoder_num_attn_heads": 6},
                "base": {"obs_encoder_hidden_dim": 768, "obs_encoder_num_layers": 12, "obs_encoder_num_attn_heads": 12},
            }
            final_params.update(standards.get(variant.lower(), {}))

        valid_keys = {f.name for f in fields(cls)}
        for k, v in yaml_data.items():
            if k in valid_keys:
                final_params[k] = v

        return cls(**final_params)
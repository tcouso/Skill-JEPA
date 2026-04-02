import torch
import yaml
from dataclasses import dataclass, fields
from typing import Tuple

@dataclass
class ModelConfig:
    # --- Identification & Metadata ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # --- LeWM Dataset ---
    dataset_name: str = "tworoom"
    frameskip: int = 5
    history_size: int = 1
    train_split: float = 0.95

    # --- Observation Encoder (Torchvision) ---
    vision_model_name: str = "vit_b_16"
    frame_size: int = 224
    num_channels: int = 3
    obs_encoder_hidden_dim: int = 768

    # --- Action Encoder (VAE) ---
    action_space_dim: int = 3
    action_sequence_length: int = 64
    action_encoder_num_layers: int = 2
    action_encoder_hidden_dim: int = 64
    action_encoder_num_attn_heads: int = 8
    
    # --- Predictor Parameters ---
    predictor_mode: str = "standard" # "standard" (AdaLN) or "jumpy" (MLP)
    decoder_num_layers: int = 4 # For ARPredictor depth
    obs_encoder_num_attn_heads: int = 12 # For ARPredictor heads (matching ViT-B)

    # --- Training ---
    batch_size: int = 40
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    max_epochs: int = 400
    warmup_epochs: int = 40

    # --- Infrastructure ---
    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    log_every_n_steps: int = 10

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)

        valid_keys = {f.name for f in fields(cls)}
        final_params = {k: v for k, v in yaml_data.items() if k in valid_keys}

        return cls(**final_params)

from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class DatasetConfig:
    env_name: str = "tworoom"
    cache_dir: str = "data"
    action_dim: int = 2
    history_size: int = 3
    pred_horizon: int = 1
    train_split: float = 0.9
    num_workers: int = 8
    frame_size: int = 224
    frameskip: int = 5
    keys_to_load: List[str] = field(default_factory=lambda: ["pixels", "action", "proprio"])
    keys_to_cache: List[str] = field(default_factory=lambda: ["action", "proprio"])


@dataclass
class VisionConfig:
    encoder_scale: str = "tiny"
    patch_size: int = 14
    embed_dim: int = 192
    proj_hidden_dim: int = 2048


@dataclass
class ActionConfig:
    space_dim: int = 2
    sequence_length: int = 2
    hidden_dim: int = 256
    num_attn_heads: int = 4
    mlp_ratio: int = 4
    num_layers: int = 2
    smoothed_dim: int = 10  # Embedder 1D-conv output dim; matches frameskip * space_dim by default


@dataclass
class PredictorConfig:
    mode: str = "standard"  # "standard" or "jumpy"
    num_layers: int = 6
    num_attn_heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 128
    base_learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    max_epochs: int = 100
    warmup_epochs: int = 10
    betas: Tuple[float, float] = (0.9, 0.95)
    log_every_n_steps: int = 100
    limit_train_batches: float = 1.0  # fraction or int; 1.0 = full dataset
    limit_val_batches: float = 1.0


@dataclass
class SIGRegConfig:
    weight: float = 0.09
    knots: int = 17
    num_proj: int = 1024


@dataclass
class EvalConfig:
    img_size: int = 224
    cache_dir: str = "data"
    dataset_name: str = "tworoom"
    goal_offset_steps: int = 25
    num_eval: int = 50
    eval_budget: int = 50
    history_size: int = 1
    frame_skip: int = 1


@dataclass
class JEPAConfig:
    seed: int = 42
    accelerator: str = "auto"
    devices: Any = "auto"
    strategy: str = "ddp"
    device: str = "cuda"
    beta_weight: float = 1.0

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Alias used in eval.py
ModelConfig = JEPAConfig

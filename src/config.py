import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any

@dataclass
class VisionConfig:
    model_name: str = "vit_b_16"
    frame_size: int = 224
    num_channels: int = 3
    hidden_dim: int = 768

@dataclass
class ActionConfig:
    space_dim: int = 3
    sequence_length: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    num_attn_heads: int = 8
    mlp_ratio: int = 4

@dataclass
class PredictorConfig:
    mode: str = "standard"  # "standard" or "jumpy"
    num_layers: int = 4
    num_attn_heads: int = 12
    dropout: float = 0.0
    emb_dropout: float = 0.0
    dim_head: int = 64
    mlp_ratio: int = 4

@dataclass
class DatasetConfig:
    name: str = "tworoom"
    frameskip: int = 5
    history_size: int = 1
    train_split: float = 0.95
    num_workers: int = 4

@dataclass
class TrainingConfig:
    batch_size: int = 40
    base_learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    max_epochs: int = 400
    warmup_epochs: int = 40
    sigreg_weight: float = 1.0
    beta_weight: float = 0.01
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    log_every_n_steps: int = 10

@dataclass
class EvalConfig:
    dataset_name: str = "tworoom"
    eval_budget: int = 100
    goal_offset_steps: int = 50
    num_eval: int = 10
    img_size: int = 224
    cache_dir: Optional[str] = None
    num_eval_episodes: int = 10

@dataclass
class SolverConfig:
    _target_: str = "stable_worldmodel.solvers.CEM" # Example default
    horizon: int = 5
    num_samples: int = 128
    iterations: int = 5
    action_dim: Optional[int] = None # Will be set at runtime

@dataclass
class WorldConfig:
    max_episode_steps: int = 200
    # Add other SWM world params here

@dataclass
class ModelConfig:
    # This is the root config object
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator: str = "gpu"
    devices: int = -1
    strategy: str = "ddp"
    
    vision: VisionConfig = field(default_factory=VisionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    world: WorldConfig = field(default_factory=WorldConfig)

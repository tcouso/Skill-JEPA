import os
from dataclasses import dataclass, field
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    dataset_path: str = "data/datasets/tworoom.h5"
    history_size: int = 3
    pred_horizon: int = 1
    train_split: float = 0.9
    num_workers: int = 8
    frame_size: int = 224
    frameskip: int = 4
    keys_to_load: list[str] = field(
        default_factory=lambda: ["pixels", "action", "proprio"]
    )
    keys_to_cache: list[str] = field(default_factory=lambda: ["action", "proprio"])


def normalized_columns(config: "JEPAConfig") -> list[str]:
    """Dataset columns that get a StandardScaler: every non-pixels key_to_load."""
    return [c for c in config.dataset.keys_to_load if not c.startswith("pixels")]


@dataclass
class VisionConfig:
    encoder_scale: str = "tiny"
    patch_size: int = 14
    embed_dim: int = 192
    proj_hidden_dim: int = 2048


@dataclass
class ProprioConfig:
    """dim=0 turns the proprio encoder off; otherwise it is the width of proprio[..., start:]."""

    dim: int = 0
    start: int = 0
    emb_dim: int = 64
    hidden_dim: int = 256


@dataclass
class ActionConfig:
    representation: str = "raw"
    space_dim: int = 2
    num_attn_heads: int = 4
    mlp_ratio: int = 4
    num_layers: int = 2
    skill_dim: int = 4
    kld_weight: float = 0.01
    normalize_kld: bool = False
    decoder_state_conditioning: bool = True
    encoder_state_conditioning: bool = False
    encoder_pos_embedding: bool = True
    detach_state_conditioning: bool = True
    detach_action_conditioning: bool = True
    detach_action_decoding: bool = False
    macro_commitment: int | None = None


@dataclass
class PredictorConfig:
    num_layers: int = 6
    num_attn_heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.0
    mlp_ratio: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 128
    reference_batch_size: int = 128
    base_learning_rate: float = 5e-5
    lr_scale_exponent: float = 0.0
    weight_decay: float = 1e-3
    max_epochs: int = 10
    warmup_fraction: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    log_every_n_steps: int = 50
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0


@dataclass
class WorldConfig:
    env_name: str = "swm/TwoRoom-v1"
    goal_conditioned: bool = True
    env_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SIGRegConfig:
    weight: float = 0.09
    knots: int = 17
    num_proj: int = 1024


@dataclass
class PLDMRegConfig:
    std_weight: float = 18.0
    std_t_weight: float = 0.7
    cov_weight: float = 12.0
    cov_t_weight: float = 0.0
    temp_align_weight: float = 0.2


@dataclass
class SMWMRegConfig:
    inverse_weight: float = 1.0
    hidden_dim: int = 256


@dataclass
class EvalConfig:
    img_size: int = 224
    goal_offset_steps: int = 20
    num_eval: int = 20
    cost_mode: str = "terminal"
    whiten_skill_space: bool = False
    sweep_goal_offsets: Any = None
    goal_cell_distance: Any = None
    sweep_goal_cell_distances: Any = None
    sweep_conditions: Any = None
    eval_tag: Any = None
    render: bool = False
    render_num_episodes: int = 10
    contact_sheet_frames: int = 10
    dataset_path: Any = None
    enabled: bool = True
    frequency: int = 5
    render_frequency: int = 10
    action_encoder_eval: bool = True
    action_spectrum_num_chunks: int = 200000
    action_recon_max_batches: Any = None
    rollout_horizon: int = 20
    state_key: str = "proprio"
    goal_state_key: str = "goal_proprio"
    state_callables: Any = None


@dataclass
class JEPAConfig:
    seed: int = 42
    accelerator: str = "auto"
    devices: Any = "auto"
    strategy: str = "ddp"
    precision: str = "bf16-mixed"
    device: str = "cuda"
    ckpt_path: Any = None
    run_id: Any = None
    checkpoint_dir: str = "./checkpoints"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    regularizer: str = "sigreg"
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    pldm: PLDMRegConfig = field(default_factory=PLDMRegConfig)
    smwm: SMWMRegConfig = field(default_factory=SMWMRegConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    solver: Any = None
    observation_encoder: Any = None
    action_encoder: Any = None


torch.serialization.add_safe_globals(
    [
        JEPAConfig,
        DatasetConfig,
        VisionConfig,
        ProprioConfig,
        ActionConfig,
        PredictorConfig,
        TrainingConfig,
        WorldConfig,
        SIGRegConfig,
        PLDMRegConfig,
        SMWMRegConfig,
        EvalConfig,
    ]
)
ConfigStore.instance().store(name="base_config", node=JEPAConfig)


def reject_stray_experiment(cfg: DictConfig) -> None:
    if "experiment" in cfg:
        raise ValueError(f"+experiment={cfg.experiment!r} matched no file in config/experiment/")


def wandb_enabled() -> bool:
    return os.environ.get("WANDB_MODE") != "disabled"

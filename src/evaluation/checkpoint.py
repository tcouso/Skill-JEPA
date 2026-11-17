"""Checkpoint loading and run setup shared by the eval entry points."""

import json
from datetime import datetime
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.config import JEPAConfig, ProprioConfig
from src.modules.jepa import SkillJEPA
from src.training import SkillJEPASystem

_EVAL_OVERRIDABLE_FIELDS = [
    ("action", "macro_commitment"),
    ("evaluation", "cost_mode"),
    ("evaluation", "whiten_skill_space"),
]


def load_system(ckpt_path: str) -> SkillJEPASystem:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = raw["hyper_parameters"]["config"]
    # Temporary patch: checkpoints saved before action.encoder_pos_embedding existed don't record it, so read it off the weights. Remove once those checkpoints are retired.
    config.action.encoder_pos_embedding = any(
        key.endswith("skill_encoder.pos_embedding") for key in raw["state_dict"]
    )
    # Checkpoints saved before the proprio section existed don't record it; they were all trained without proprio.
    if not hasattr(config, "proprio"):
        config.proprio = ProprioConfig()
    system = SkillJEPASystem(config)
    system.load_state_dict(raw["state_dict"])
    return system


def apply_eval_overrides(system: SkillJEPASystem, config: JEPAConfig) -> None:
    """Copy this run's eval knobs onto the checkpoint's own config, which the model reads."""
    for section, field in _EVAL_OVERRIDABLE_FIELDS:
        setattr(getattr(system.jepa.config, section), field, getattr(getattr(config, section), field))


def prepare_system(cfg: DictConfig, config: JEPAConfig) -> SkillJEPASystem:
    """Load the checkpoint for eval, refusing a mismatched env or an unfitted skill prior."""
    system = load_system(cfg.ckpt_path).to(config.device).eval()
    system.requires_grad_(False)

    # Same-frameskip checkpoints load into either env: TwoRoom and PushT state dicts share shapes.
    trained_on = system.jepa.config.world.env_name
    if trained_on != config.world.env_name:
        raise ValueError(f"checkpoint trained on {trained_on!r}, run configured for {config.world.env_name!r}")

    if isinstance(system.jepa, SkillJEPA):
        bottleneck = system.jepa.skill_bottleneck
        if not bottleneck.skill_prior_mu.any() and bool((bottleneck.skill_prior_var == 1).all()):
            raise ValueError("skill_prior_mu/var are unfitted; retrain so on_train_epoch_end fits them")

    apply_eval_overrides(system, config)
    return system


def derive_tag(config: JEPAConfig) -> str:
    """Run label: explicit eval_tag, else the +experiment choice, else arm+frameskip."""
    experiment_name = HydraConfig.get().runtime.choices.get("experiment", None)
    return (
        config.evaluation.eval_tag
        or experiment_name
        or f"{config.action.representation}_fs{config.dataset.frameskip}"
    )


def results_dir(tag: str) -> Path:
    path = Path("results") / "sweeps" / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_info(directory: Path, cfg: DictConfig, train_seed: int) -> None:
    info = {
        **OmegaConf.to_container(cfg, resolve=True),
        "train_seed": train_seed,
        "created_at": datetime.now().isoformat(),
    }
    (directory / "run.json").write_text(json.dumps(info, indent=2))

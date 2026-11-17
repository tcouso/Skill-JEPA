"""Component evals of one checkpoint, no planning: action reconstruction, and prediction error along open-loop rollouts against the frames actually reached, on dataset plans and on CEM's first-iteration candidates."""

import dataclasses
import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import stable_worldmodel as swm
import torch
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.policy import PlanConfig

from src.config import JEPAConfig, reject_stray_experiment, wandb_enabled
from src.evaluation.checkpoint import derive_tag, prepare_system, write_run_info
from src.evaluation.metrics.action_encoder import floor_mse, reconstruction_metrics
from src.evaluation.mpc_eval import (
    _state_callables,
    build_process,
    episode_col,
    get_dataset,
    sample_eval_episodes,
)
from src.modules.action import VAEBottleneck
from src.modules.jepa import SkillJEPA
from src.policy import PlanDecodingPolicy
from src.training.datamodule import DataModule
from src.training.transforms import build_image_transform


def collect_finite_action_chunks(
    dataloader, frameskip: int, max_samples: int
) -> tuple[torch.Tensor, int]:
    """Action chunks without non-finite steps, plus how many were dropped."""
    # Each episode's terminal step carries a non-finite action.
    kept, dropped, total = [], 0, 0
    for batch in dataloader:
        chunks = rearrange(batch["action"], "b t (f d) -> (b t) f d", f=frameskip)
        finite = torch.isfinite(chunks).all(dim=-1).all(dim=-1)
        dropped += int((~finite).sum())
        kept.append(chunks[finite])
        total += int(finite.sum())
        if total >= max_samples:
            break

    return torch.cat(kept, dim=0)[:max_samples], dropped


@torch.no_grad()
def collect_action_reconstructions(
    jepa: SkillJEPA, dataloader, device: str, max_batches: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(decode(encode(chunk)), chunk) pooled over `dataloader`."""
    recons, targets = [], []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        pixels = batch["pixels"].to(device)
        action = torch.nan_to_num(batch["action"].to(device), 0.0)
        obs_emb = jepa.encode_obs(pixels, batch.get("proprio"))
        act_output = jepa.encode_act(action, obs_emb)
        recon = jepa.decode_act(act_output["skill"], obs_emb)

        recons.append(rearrange(recon, "b t f d -> (b t) f d").cpu())
        targets.append(rearrange(act_output["act_chunk"], "b t f d -> (b t) f d").cpu())

    return torch.cat(recons, dim=0), torch.cat(targets, dim=0)


def evaluate_action_encoder(jepa, config: JEPAConfig) -> dict:
    """The spectrum floor on every arm; reconstruction on latent arms."""
    datamodule = DataModule(config)
    datamodule.setup()

    chunks, dropped = collect_finite_action_chunks(
        datamodule.train_dataloader(),
        frameskip=config.dataset.frameskip,
        max_samples=config.evaluation.action_spectrum_num_chunks,
    )
    metrics = {
        "action_spectrum/floor_mse_at_skill_dim": floor_mse(chunks, config.action.skill_dim),
        "action_spectrum/n_chunks": chunks.shape[0],
        "action_spectrum/n_dropped": dropped,
    }
    if not isinstance(jepa, SkillJEPA):
        return metrics

    recon, target = collect_action_reconstructions(
        jepa,
        datamodule.val_dataloader(),
        device=config.device,
        max_batches=config.evaluation.action_recon_max_batches,
    )
    metrics.update(reconstruction_metrics(recon, target))
    return metrics


class FrameRecorder:
    """Passthrough policy keeping each env's first `n_frames` frames, one every `frameskip` steps, flagging the ones after it terminated."""

    def __init__(self, policy, n_envs: int, frameskip: int, n_frames: int) -> None:
        self._policy = policy
        self._frameskip = frameskip
        self._n_frames = n_frames
        self._step = 0
        self._terminal: list[np.ndarray | None] = [None] * n_envs
        self.frames: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        self.frozen: list[list[bool]] = [[] for _ in range(n_envs)]

    def get_action(self, obs):
        on_macro_step = self._step % self._frameskip == 0 and self._step // self._frameskip < self._n_frames
        self.observe(obs["pixels"], obs["terminated"], on_macro_step)
        self._step += 1
        return self._policy.get_action(obs)

    def observe(self, pixels, terminated, on_macro_step: bool) -> None:
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        for i in range(len(self.frames)):
            if terminated[i] and self._terminal[i] is None:
                self._terminal[i] = self._last_frame(pixels[i])
            if not on_macro_step:
                continue
            frozen = self._terminal[i] is not None
            self.frames[i].append(self._terminal[i] if frozen else self._last_frame(pixels[i]))
            self.frozen[i].append(frozen)

    def finish(self, pixels, terminated) -> None:
        """Fill the remaining frames from the world's last observation; the world stops early once all envs terminated."""
        while len(self.frames[0]) < self._n_frames:
            self.observe(pixels, terminated, on_macro_step=True)

    @staticmethod
    def _last_frame(pixels) -> np.ndarray:
        frame = np.asarray(pixels)
        return (frame[-1] if frame.ndim > 3 else frame).copy()

    def set_env(self, env) -> None:
        self._policy.set_env(env)

    def set_seed(self, seed: int) -> None:
        self._policy.set_seed(seed)


def hold_after_termination(errors: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
    """Each (n, steps) error keeps its last value from before the env terminated."""
    errors = errors.clone()
    for k in range(1, errors.size(1)):
        errors[:, k] = torch.where(frozen[:, k], errors[:, k - 1], errors[:, k])
    return errors


@torch.no_grad()
def encode_frames(jepa, frames: list[list[np.ndarray]], img_transform, device: str) -> torch.Tensor:
    """(n, steps, d) embeddings of each env's frames."""
    return torch.stack([
        jepa.encode_obs(torch.stack([img_transform(frame) for frame in env_frames])[None].to(device))[0].cpu()
        for env_frames in frames
    ])


def rollout_conditioning(jepa, action: torch.Tensor, obs_emb: torch.Tensor) -> torch.Tensor:
    """What the planner feeds rollout: raw action chunks, or the deterministic skill of each dataset chunk."""
    if not isinstance(jepa, SkillJEPA):
        return action
    act_output = jepa.encode_act(action, obs_emb)
    if isinstance(jepa.skill_bottleneck, VAEBottleneck):
        return act_output["skill_mu"]
    return act_output["skill"]


def load_windows(
    dataset, episodes, start_steps, horizon: int, frameskip: int, process: dict, img_transform
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per window: frames every `frameskip` steps (n, horizon + 1, C, H, W) and normalised action chunks (n, horizon, frameskip * d)."""
    ep_col = dataset.get_col_data(episode_col(dataset))
    step_col = dataset.get_col_data("step_idx")
    pixels, actions = [], []
    for ep, start in zip(episodes, start_steps):
        first = int(np.flatnonzero((ep_col == ep) & (step_col == start))[0])
        frames = dataset.get_row_data(first + frameskip * np.arange(horizon + 1))["pixels"]
        pixels.append(torch.stack([img_transform(frame) for frame in frames]))
        action = dataset.get_row_data(first + np.arange(frameskip * horizon))["action"]
        if "action" in process:
            action = process["action"].transform(action)
        actions.append(torch.as_tensor(action, dtype=torch.float32).reshape(horizon, -1))

    return torch.stack(pixels), torch.stack(actions)


@torch.no_grad()
def rollout_windows(
    jepa, pixels: torch.Tensor, actions: torch.Tensor, device: str, chunk_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """(predicted embeddings from the first frame (n, horizon + 1, d), plans in CEM's coordinates (n, horizon, plan_dim))."""
    horizon = actions.size(1)
    predicted, plans = [], []
    for i in range(0, pixels.size(0), chunk_size):
        px = pixels[i : i + chunk_size].to(device)
        act = actions[i : i + chunk_size].to(device)
        obs_emb = jepa.encode_obs(px)
        plan = rollout_conditioning(jepa, act, obs_emb[:, :horizon])
        predicted_emb, _ = jepa.rollout(px[:, None, :1], None, plan[:, None])

        predicted.append(predicted_emb[:, 0].cpu())
        plans.append(plan.cpu())

    return torch.cat(predicted), torch.cat(plans)


def sample_iteration0_plans(jepa, config: JEPAConfig, n: int, horizon: int) -> torch.Tensor:
    """One of CEM's first-iteration candidates per window, in CEM's coordinates (n, horizon, plan_dim)."""
    generator = torch.Generator().manual_seed(config.seed)
    # CEMSolver multiplies its unit-normal samples by var_scale, so var_scale is the std.
    return config.solver["var_scale"] * torch.randn(n, horizon, jepa.plan_dim, generator=generator)


@torch.no_grad()
def rollout_plans(jepa, pixels: torch.Tensor, plans: torch.Tensor, device: str, chunk_size: int = 16) -> torch.Tensor:
    """Predicted embeddings from each window's first frame (n, horizon + 1, d) under plans in CEM's coordinates."""
    predicted = []
    for i in range(0, pixels.size(0), chunk_size):
        px = pixels[i : i + chunk_size, :1].to(device)
        plan = jepa._from_search_space(plans[i : i + chunk_size].to(device))
        predicted_emb, _ = jepa.rollout(px[:, None], None, plan[:, None])
        predicted.append(predicted_emb[:, 0].cpu())

    return torch.cat(predicted)


class ReplaySolver:
    """Stands in for CEM: hands every env its precomputed plan in one call."""

    def __init__(self, plans: torch.Tensor) -> None:
        self.plans = plans

    def configure(self, action_space, n_envs: int, config) -> None:
        if n_envs != self.plans.size(0):
            raise ValueError(f"{self.plans.size(0)} plans for {n_envs} envs")

    def __call__(self, info_dict: dict) -> dict:
        if info_dict["pixels"].size(0) != self.plans.size(0):
            raise RuntimeError("ReplaySolver called for a subset of envs; it only replays the first plan")
        return {"actions": self.plans, "costs": torch.zeros(self.plans.size(0))}


def execute_windows(
    jepa, config: JEPAConfig, dataset, episodes, start_steps, plans: torch.Tensor, process: dict, img_transform
) -> tuple[list[list[np.ndarray]], torch.Tensor]:
    """Execute decode(plan) open loop from each window's start state; (frames per env, frozen mask (n, horizon + 1))."""
    horizon = plans.size(1)
    budget = horizon * config.dataset.frameskip
    world_cfg = dataclasses.asdict(config.world)
    env_kwargs = world_cfg.pop("env_kwargs")
    world_cfg["max_episode_steps"] = 2 * budget
    world_cfg["num_envs"] = len(episodes)
    world = swm.World(
        **world_cfg,
        **env_kwargs,
        image_shape=dataset.get_row_data(0)["pixels"].shape[:2],
    )

    policy = PlanDecodingPolicy(
        solver=ReplaySolver(plans.to(config.device)),
        jepa=jepa,
        process=process,
        transform={"pixels": img_transform, "goal": img_transform},
        plan_config=PlanConfig(
            horizon=horizon,
            receding_horizon=horizon,
            history_len=config.dataset.history_size,
            action_block=1,
        ),
        device=config.device,
    )
    recorder = FrameRecorder(policy, len(episodes), config.dataset.frameskip, n_frames=horizon + 1)
    world.set_policy(recorder)

    with torch.no_grad():
        world.evaluate(
            dataset=dataset,
            episodes_idx=episodes.tolist(),
            start_steps=start_steps.tolist(),
            goal_offset=budget,
            eval_budget=budget,
            callables=_state_callables(config),
        )
    # The world asks for an action before each step, so the frame after the last one is only in world.infos.
    recorder.finish(world.infos["pixels"], world.terminateds)
    del world

    return recorder.frames, torch.tensor(recorder.frozen)


def squared_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean(dim=-1)


def error_by_step(predicted: torch.Tensor, executed: torch.Tensor, frozen: torch.Tensor) -> pd.DataFrame:
    """Per step, predicted-vs-executed embedding error over the persistence baseline (executed first frame vs executed)."""
    plan_mse = hold_after_termination(squared_error(predicted, executed), frozen)[:, 1:].mean(dim=0)
    copy_mse = hold_after_termination(squared_error(executed[:, :1], executed), frozen)[:, 1:].mean(dim=0)
    return pd.DataFrame({
        "step": np.arange(1, plan_mse.size(0) + 1),
        "plan_execution_mse": plan_mse.numpy(),
        "executed_copy_mse": copy_mse.numpy(),
        "plan_execution_over_copy": (plan_mse / copy_mse).numpy(),
        "n_frozen": frozen[:, 1:].sum(dim=0).numpy(),
    })


def evaluate_plans(jepa, config: JEPAConfig, source: str) -> pd.DataFrame:
    """Error by step on dataset windows, for the dataset's own plans ("dataset") or CEM first-iteration candidates ("iteration0")."""
    horizon = config.evaluation.rollout_horizon
    frameskip = config.dataset.frameskip
    window_config = dataclasses.replace(
        config,
        evaluation=dataclasses.replace(config.evaluation, goal_offset_steps=horizon * frameskip),
    )
    dataset = get_dataset(config, config.evaluation.dataset_path or config.dataset.dataset_path)
    process = build_process(dataset, config)
    img_transform = build_image_transform(config.evaluation.img_size)
    episodes, start_steps = sample_eval_episodes(dataset, window_config, config.seed)

    pixels, actions = load_windows(dataset, episodes, start_steps, horizon, frameskip, process, img_transform)
    if source == "dataset":
        predicted, plans = rollout_windows(jepa, pixels, actions, config.device)
    else:
        plans = sample_iteration0_plans(jepa, config, len(episodes), horizon)
        predicted = rollout_plans(jepa, pixels, plans, config.device)
    frames, frozen = execute_windows(jepa, config, dataset, episodes, start_steps, plans, process, img_transform)
    executed = encode_frames(jepa, frames, img_transform, config.device)
    table = error_by_step(predicted, executed, frozen)
    print(f"[{source}]\n{table.to_string(index=False)}")
    return table


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    reject_stray_experiment(cfg)
    config: JEPAConfig = OmegaConf.to_object(cfg)
    system = prepare_system(cfg, config)

    train_seed = system.jepa.config.seed
    tag = f"{derive_tag(config)}_seed{train_seed}"
    metrics = evaluate_action_encoder(system.jepa, config) if config.evaluation.action_encoder_eval else {}
    print(metrics)
    tables = {source: evaluate_plans(system.jepa, config, source) for source in ("dataset", "iteration0")}
    if wandb_enabled():
        wandb.init(
            project="SkillJEPA_Eval",
            name=f"Components_{tag}",
            config={**OmegaConf.to_container(cfg, resolve=True), "train_seed": train_seed},
        )
        wandb.log({**metrics, **{f"{source}/error_by_step": wandb.Table(dataframe=table) for source, table in tables.items()}})
        wandb.finish()
        return
    out = Path("results") / "components" / tag
    out.mkdir(parents=True, exist_ok=True)
    write_run_info(out, cfg, train_seed)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for source, table in tables.items():
        table.to_csv(out / f"{source}_error_by_step.csv", index=False)


if __name__ == "__main__":
    run()

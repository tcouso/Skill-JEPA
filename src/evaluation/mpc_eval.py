"""One MPC eval of one cell: build the world and policy, run it, aggregate metrics."""

import dataclasses
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from sklearn import preprocessing
from stable_worldmodel.policy import PlanConfig

import src.envs  # noqa: F401
from src.config import JEPAConfig, normalized_columns
from src.envs.maze_cells import goal_at_cell_distance
from src.modules.jepa import BaseJEPA
from src.policy import PlanDecodingPolicy
from src.training.transforms import build_image_transform
from src.viz.contact_sheet import make_contact_sheet


def run_mpc_eval(
    jepa: BaseJEPA,
    config: JEPAConfig,
    cfg: DictConfig,
    render: bool = False,
) -> tuple[dict, list[dict], list]:
    """Returns (scalar metrics, one row per episode, contact sheets)."""
    budget = eval_budget(config)
    world_cfg = dataclasses.asdict(config.world)
    env_kwargs = world_cfg.pop("env_kwargs")
    world_cfg["max_episode_steps"] = 2 * budget

    img_transform = build_image_transform(config.evaluation.img_size)
    transform = {"pixels": img_transform, "goal": img_transform}
    dataset = get_dataset(config, config.evaluation.dataset_path or config.dataset.dataset_path)
    process = build_process(dataset, config)
    episodes, start_steps = sample_eval_episodes(dataset, config, config.seed)

    world_cfg["num_envs"] = len(episodes)
    world = swm.World(
        **world_cfg,
        **env_kwargs,
        # World overwrites its pixels/goal buffers with dataset frames, so env frames must match their size.
        image_shape=dataset.get_row_data(0)["pixels"].shape[:2],
    )
    policy = build_policy(jepa, config, cfg, process, transform)
    n_render = min(config.evaluation.render_num_episodes, len(episodes)) if render else 0
    tracker = TrackingPolicy(policy, len(episodes), capture_envs=n_render)
    world.set_policy(tracker)

    with torch.no_grad():
        results = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes.tolist(),
            start_steps=start_steps.tolist(),
            goal_offset=config.evaluation.goal_offset_steps,
            eval_budget=budget,
            callables=_state_callables(config),
        )
    del world

    successes = results["episode_successes"]
    tracker.finalize(successes)
    metrics = {"success_rate": results["success_rate"], "n_success": int(sum(successes))}
    rows = _episode_rows(episodes, start_steps, successes, tracker.steps_to_done)
    contact_sheets = [
        make_contact_sheet(
            tracker.frames[i],
            tracker.goal_frames[i],
            max_frames=config.evaluation.contact_sheet_frames,
            success=bool(successes[i]),
            total_steps=budget,
        )
        for i in range(n_render)
    ]
    return metrics, rows, contact_sheets


def get_dataset(config: JEPAConfig, dataset_path: str):
    p = Path(dataset_path).resolve()
    return swm.data.load_dataset(
        str(p),
        cache_dir=str(p.parent.parent),
        keys_to_cache=list(config.dataset.keys_to_cache),
    )


def episode_col(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def get_episodes_length(dataset, episodes: np.ndarray) -> np.ndarray:
    col = episode_col(dataset)
    episode_idx = dataset.get_col_data(col)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def build_process(dataset, config: JEPAConfig) -> dict:
    """Per-column StandardScalers over the columns training normalizes."""
    process = {}
    for col in normalized_columns(config):
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def sample_eval_episodes(dataset, config: JEPAConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(episode_ids, start_steps), each start leaving goal_offset_steps before its episode ends."""
    col = episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)

    goal_offset = config.evaluation.goal_offset_steps
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = dict(zip(ep_indices, max_start_idx, strict=True))
    max_start_per_row = np.array(
        [max_start_idx_dict[ep] for ep in dataset.get_col_data(col)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    if config.evaluation.goal_cell_distance is not None:
        valid_mask &= goal_at_cell_distance(dataset, config)
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid_indices), size=config.evaluation.num_eval, replace=False)
    chosen = np.sort(valid_indices[chosen])

    episodes = dataset.get_row_data(chosen)[col]
    start_steps = dataset.get_row_data(chosen)["step_idx"]
    return episodes, start_steps


def plan_horizon(config: JEPAConfig) -> int:
    """Macro-steps in one plan; one plan exactly spans the goal offset."""
    offset, frameskip = config.evaluation.goal_offset_steps, config.dataset.frameskip
    if offset % frameskip:
        raise ValueError(f"goal_offset_steps={offset} is not a multiple of frameskip={frameskip}")
    return offset // frameskip


def eval_budget(config: JEPAConfig) -> int:
    return 2 * config.evaluation.goal_offset_steps


def _state_callables(config: JEPAConfig) -> list[dict]:
    """Seed the physics sim from the dataset episode instead of a random reset."""
    if config.evaluation.state_callables is not None:
        return config.evaluation.state_callables
    return [
        {"method": "_set_state", "args": {"state": {"value": config.evaluation.state_key}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": config.evaluation.goal_state_key}}},
    ]


def build_policy(
    jepa: BaseJEPA,
    config: JEPAConfig,
    cfg: DictConfig,
    process: dict,
    transform: dict,
) -> PlanDecodingPolicy:
    horizon = plan_horizon(config)
    macro_commitment = config.action.macro_commitment
    plan_config = PlanConfig(
        horizon=horizon,
        receding_horizon=horizon if macro_commitment is None else macro_commitment,
        history_len=config.dataset.history_size,
        # The solver multiplies the Box width (already plan_dim) by action_block.
        action_block=1,
    )
    solver = hydra.utils.instantiate(cfg.solver, model=jepa)

    return PlanDecodingPolicy(
        solver=solver,
        jepa=jepa,
        process=process,
        transform=transform,
        plan_config=plan_config,
        device=config.device,
    )


class TrackingPolicy:
    """Passthrough policy recording termination steps, and frames for the first `capture_envs` envs."""

    def __init__(self, policy, n_envs: int, capture_envs: int = 0) -> None:
        self._policy = policy
        self._capture_envs = min(capture_envs, n_envs)
        self.frames: list[list] = [[] for _ in range(self._capture_envs)]
        self.goal_frames: list[np.ndarray | None] = [None] * self._capture_envs
        self._done = np.zeros(n_envs, dtype=bool)
        self._step = 0
        self.steps_to_done: list[int | None] = [None] * n_envs

    def get_action(self, obs):
        for i in range(self._capture_envs):
            if not self._done[i]:
                self.frames[i].append(np.array(obs["pixels"][i]))
            if self.goal_frames[i] is None:
                self.goal_frames[i] = np.array(obs["goal"][i])

        terminated = np.asarray(obs["terminated"]).reshape(-1)
        for i in np.flatnonzero(terminated & ~self._done):
            self.steps_to_done[i] = self._step
        self._done |= terminated
        self._step += 1

        return self._policy.get_action(obs)

    def finalize(self, episode_successes) -> None:
        """Backfills terminations the action loop could not observe."""
        # The world calls get_action before each step, so it never sees a last-step termination.
        for i in np.flatnonzero(episode_successes):
            if self.steps_to_done[i] is None:
                self.steps_to_done[i] = self._step

    def set_env(self, env) -> None:
        self._policy.set_env(env)

    def set_seed(self, seed: int) -> None:
        self._policy.set_seed(seed)


def _episode_rows(episodes, start_steps, successes, steps_to_done) -> list[dict]:
    return [
        {
            "env_index": i,
            "dataset_episode": int(episodes[i]),
            "start_step": int(start_steps[i]),
            "success": bool(successes[i]),
            "steps_to_done": steps_to_done[i],
        }
        for i in range(len(episodes))
    ]

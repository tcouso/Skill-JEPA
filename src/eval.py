import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from src.system import ModelSystem
from src.config import ModelConfig

# TODO: We need an evaluation that counts the planning steps at inference
# Our selling point will be to plan with comparable results using less steps


def img_transform(cfg: ModelConfig):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg: ModelConfig, dataset_name):
    dataset_path = Path(cfg.eval.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=["action", "proprio"],  # Example keys
        cache_dir=dataset_path,
    )
    return dataset


class AgnosticBufferedPolicy:
    def __init__(self, solver, jepa: torch.nn.Module, process: dict, transform: dict):
        self.solver = solver
        self.jepa = jepa
        self.process = process
        self.transform = transform
        self.action_buffer = []

    def get_action(self, obs: dict):
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)

        info_dict = {}
        for k, v in obs.items():
            if k in self.transform:
                v = self.transform[k](v)
            if k in self.process:
                v = torch.tensor(self.process[k].transform(np.array([v])))[0]
            if isinstance(v, torch.Tensor):
                info_dict[k] = v.unsqueeze(0).to(next(self.jepa.parameters()).device)
            else:
                info_dict[k] = (
                    torch.tensor(v).unsqueeze(0).to(next(self.jepa.parameters()).device)
                )

        best_sequence = self.solver(info_dict)
        chunk_tensor = self.jepa.get_executable_actions(info_dict, best_sequence)

        chunk_np = chunk_tensor.cpu().numpy()
        if "action" in self.process:
            chunk_np = self.process["action"].inverse_transform(chunk_np)

        self.action_buffer = list(chunk_np)
        return self.action_buffer.pop(0)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(config: ModelConfig):
    ckpt_path = OmegaConf.to_container(config).get("ckpt_path", None)
    if not ckpt_path:
        print(
            "Warning: No ckpt_path provided. Using randomly initialized model (for testing only)."
        )
        system = ModelSystem(config)
    else:
        system = ModelSystem.load_from_checkpoint(ckpt_path)

    system = system.to(config.device).eval()
    system.requires_grad_(False)
    jepa = system.jepa

    run_name = f"Eval_{config.predictor.mode}_{config.eval.dataset_name}"
    wandb.init(
        project="SkillJEPA_Eval",
        name=run_name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    world_cfg = OmegaConf.to_container(config.world, resolve=True)
    world = swm.World(
        **world_cfg, image_shape=(config.vision.frame_size, config.vision.frame_size)
    )

    transform = {
        "pixels": img_transform(config),
        "goal": img_transform(config),
    }

    dataset = get_dataset(config, config.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    keys_to_process = ["action", "proprio"]
    for col in keys_to_process:
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    if config.predictor.mode == "jumpy":
        config.solver.action_dim = config.action.hidden_dim
    else:
        config.solver.action_dim = config.action.space_dim

    solver = hydra.utils.instantiate(config.solver, model=jepa)
    policy = AgnosticBufferedPolicy(
        solver=solver, jepa=jepa, process=process, transform=transform
    )

    results_path = Path(__file__).parent.parent / "results" / config.predictor.mode
    results_path.mkdir(parents=True, exist_ok=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - config.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(config.seed)
    num_eval = min(config.eval.num_eval, len(valid_indices))
    random_indices = g.choice(len(valid_indices), size=num_eval, replace=False)
    random_indices = np.sort(valid_indices[random_indices])

    eval_episodes = dataset.get_row_data(random_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_indices)["step_idx"]

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=config.eval.goal_offset_steps,
        eval_budget=config.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        video_path=results_path,
    )
    end_time = time.time()

    metrics["eval_time"] = end_time - start_time
    print(metrics)

    wandb.log(metrics)
    wandb.finish()

    output_file = results_path / "results.txt"
    with output_file.open("a") as f:
        f.write(f"\n==== Eval Results {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        f.write(f"Metrics: {metrics}\n")


if __name__ == "__main__":
    run()

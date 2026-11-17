"""Derives AntMaze's sweep_goal_offsets in config/env/antmaze.yaml; rerun by hand if the dataset changes.

    uv run python src/antmaze_goal_offsets.py +experiment=lewm/raw_antmaze

Each offset is the median steps from a random start to first reach that maze-cell distance,
rounded up to a multiple of frameskip.
"""

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.config import JEPAConfig, reject_stray_experiment
from src.envs.maze_cells import maze_cells
from src.evaluation.mpc_eval import episode_col, get_dataset


def cell_distance_offsets(
    dataset, config: JEPAConfig, cell_distances: list[int], num_starts: int = 200_000, seed: int = 0
) -> list[int]:
    i, j, distances = maze_cells(dataset, config)
    _, first_rows, lengths = np.unique(
        dataset.get_col_data(episode_col(dataset)), return_index=True, return_counts=True
    )

    rng = np.random.default_rng(seed)
    steps_to = {d: [] for d in cell_distances}
    for episode in rng.integers(len(first_rows), size=num_starts):
        start = first_rows[episode] + rng.integers(lengths[episode] - 1)
        end = first_rows[episode] + lengths[episode]
        from_start = distances[i[start], j[start], i[start:end], j[start:end]]
        for d in cell_distances:
            reached = np.flatnonzero(from_start == d)
            if len(reached):
                steps_to[d].append(reached[0])

    frameskip = config.dataset.frameskip
    return [int(np.ceil(np.median(steps_to[d]) / frameskip)) * frameskip for d in cell_distances]


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    reject_stray_experiment(cfg)
    config: JEPAConfig = OmegaConf.to_object(cfg)
    dataset = get_dataset(config, config.evaluation.dataset_path or config.dataset.dataset_path)
    cell_distances = list(config.evaluation.sweep_goal_cell_distances)
    offsets = cell_distance_offsets(dataset, config, cell_distances)
    print(f"sweep_goal_cell_distances: {cell_distances}")
    print(f"sweep_goal_offsets: {offsets}")


if __name__ == "__main__":
    run()

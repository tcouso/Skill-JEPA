"""One reset frame per task for the paper: the four detail-analysis tasks side by side, and AntMaze on its own.

Usage:
    MUJOCO_GL=egl uv run python src/viz/task_frames.py
"""

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel  # noqa: F401

import src.envs  # noqa: F401
from src.viz.sweep_plots import ENV_NAMES

OUTPUT_DIR = Path("results/task_frames")
SEED = 0
SIZE = 224
PANEL_W = 1.35

TASKS = {
    "cube": ("swm/OGBCube-v0", dict(env_type="single", ob_type="states", multiview=False, width=SIZE, height=SIZE,
                                         visualize_info=False)),
    "pusht": ("swm/PushT-v1", {}),
    "reacher": ("swm/ReacherDMControl-v0", dict(task="qpos_match")),
    "tworoom": ("swm/TwoRoom-v1", {}),
    "antmaze": ("swm/OGBAntMaze-v0",
                dict(maze_type="medium", ob_type="pixels", width=SIZE, height=SIZE, camera_name="back")),
}


def task_frame(name: str) -> np.ndarray:
    env_id, kwargs = TASKS[name]
    env = gym.make(env_id, render_mode="rgb_array", **kwargs)
    obs, _ = env.reset(seed=SEED)
    # AntMaze's render() returns a black frame; its pixel observation is the back camera.
    if name == "antmaze":
        return np.asarray(obs)
    return np.asarray(env.render())


def save_frames(names: list[str], path: Path, titled: bool) -> None:
    height = PANEL_W + 0.25 if titled else PANEL_W
    fig, axes = plt.subplots(1, len(names), figsize=(PANEL_W * len(names), height), squeeze=False)
    for ax, name in zip(axes[0], names):
        ax.imshow(task_frame(name))
        if titled:
            ax.set_title(ENV_NAMES[name], fontsize=12)
        ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {path} and {path.with_suffix('.png')}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_frames(["cube", "pusht", "reacher", "tworoom"], OUTPUT_DIR / "task_frames.pdf", titled=True)
    save_frames(["antmaze"], OUTPUT_DIR / "antmaze_frame.pdf", titled=False)


if __name__ == "__main__":
    main()

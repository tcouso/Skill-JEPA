"""Maze-cell geometry for the OGBench mazes: which cell a state is in, and shortest paths between cells."""

from collections import deque

import gymnasium as gym
import numpy as np

from src.config import JEPAConfig


def maze_cell_distances(maze: np.ndarray) -> np.ndarray:
    """[i, j, i2, j2] -> shortest path in cells between two open cells; -1 where either is a wall."""
    rows, cols = maze.shape
    distances = np.full((rows, cols, rows, cols), -1)
    for start in zip(*np.nonzero(maze == 0)):
        from_start = distances[start]
        from_start[start] = 0
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if maze[ni, nj] == 0 and from_start[ni, nj] < 0:
                    from_start[ni, nj] = from_start[i, j] + 1
                    queue.append((ni, nj))
    return distances


def maze_cells(dataset, config: JEPAConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each row's maze cell (i, j), and the env maze's cell-distance table."""
    env = gym.make(config.world.env_name, **config.world.env_kwargs)
    maze = env.unwrapped
    # OGBench's MazeEnv.xy_to_ij, vectorised over rows; it keeps unit and offsets private.
    unit = maze._maze_unit
    xy = dataset.get_col_data("xy")
    i = ((xy[:, 1] + maze._offset_y + unit / 2) // unit).astype(int)
    j = ((xy[:, 0] + maze._offset_x + unit / 2) // unit).astype(int)
    distances = maze_cell_distances(np.asarray(maze.maze_map))
    env.close()
    return i, j, distances


def goal_at_cell_distance(dataset, config: JEPAConfig) -> np.ndarray:
    """Per row: is the state goal_offset_steps later exactly goal_cell_distance maze cells away."""
    i, j, distances = maze_cells(dataset, config)
    offset = config.evaluation.goal_offset_steps
    distance = np.full(len(i), -1)
    distance[:-offset] = distances[i[:-offset], j[:-offset], i[offset:], j[offset:]]
    return distance == config.evaluation.goal_cell_distance

"""Per-env action-buffer bookkeeping, shared by both representations' replan loop."""

from collections import deque

import numpy as np
import torch


class ReplanState:
    """Per-env buffers of decoded actions still to execute."""

    def __init__(self, n_envs: int):
        self._buffers: list[deque] = [deque() for _ in range(n_envs)]

    def begin_step(self, terminated, needs_flush) -> tuple[np.ndarray, list[int]]:
        """Returns (dead, replan_idx), first flushing envs the world reset under us."""
        n_envs = len(self._buffers)
        dead = self._as_bool_mask(terminated, n_envs)

        if needs_flush is not None:
            flush = self._as_bool_mask(needs_flush, n_envs)
            for i in range(n_envs):
                if flush[i]:
                    self._buffers[i].clear()

        replan_idx = [
            i for i in range(n_envs) if not dead[i] and len(self._buffers[i]) == 0
        ]
        return dead, replan_idx

    @staticmethod
    def _as_bool_mask(value, n_envs: int) -> np.ndarray:
        if value is None:
            return np.zeros(n_envs, dtype=bool)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=bool).reshape(-1)

    def commit_plan(self, replan_idx: list[int], decoded_actions: np.ndarray) -> None:
        """Push `decoded_actions` (n_replan, T, action_dim) onto each replanning env's buffer."""
        for row, env_i in enumerate(replan_idx):
            for t in range(decoded_actions.shape[1]):
                self._buffers[env_i].append(decoded_actions[row, t, :])

    def pop_actions(self, dead: np.ndarray, action_dim: int) -> np.ndarray:
        """One physical action per alive env; dead envs get a discarded placeholder."""
        n_envs = len(self._buffers)
        action = np.zeros((n_envs, action_dim), dtype=np.float32)
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._buffers[i].popleft()
        return action

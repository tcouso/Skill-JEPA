import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.policy import BasePolicy, PlanConfig

from src.modules.jepa import BaseJEPA
from src.policy.replan_state import ReplanState


class PlanDecodingPolicy(BasePolicy):
    """World-model policy whose CEM output needs decoding into physical actions."""

    def __init__(
        self,
        solver,
        jepa: BaseJEPA,
        process: dict,
        transform: dict,
        plan_config: PlanConfig,
        device: str,
    ):
        super().__init__()
        self.solver = solver
        self.jepa = jepa
        self.process = process
        self.transform = transform
        self._plan_cfg = plan_config
        self.device = device
        self._state: ReplanState | None = None

    def set_env(self, env) -> None:
        """Configure the solver for the given vectorised environment."""
        self.env = env
        n_envs = getattr(env, "num_envs", 1)

        plan_dim = self.jepa.plan_dim
        action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_envs, plan_dim),
            dtype=np.float32,
        )

        self.solver.configure(
            action_space=action_space, n_envs=n_envs, config=self._plan_cfg
        )

        self._state = ReplanState(n_envs)

    def get_action(self, obs: dict) -> np.ndarray:
        info_dict = self._prepare(obs)
        needs_flush = info_dict.pop("_needs_flush", None)
        dead, replan_idx = self._state.begin_step(info_dict.get("terminated"), needs_flush)

        if replan_idx:
            self._replan(info_dict, replan_idx)

        return self._state.pop_actions(dead, self.jepa.config.action.space_dim)

    def _prepare(self, obs: dict) -> dict:
        """Normalize the raw env observation and move tensors onto self.device."""
        info_dict = self._prepare_info(dict(obs))
        for key, val in info_dict.items():
            if torch.is_tensor(val):
                info_dict[key] = val.to(self.device)

        return info_dict

    def _replan(self, info_dict: dict, replan_idx: list[int]) -> None:
        """Solve for `replan_idx` envs and push their committed plan onto ReplanState."""
        replan_idx_tensor = torch.as_tensor(
            replan_idx, dtype=torch.long, device=self.device
        )
        replan_info = {}

        for k, v in info_dict.items():
            if torch.is_tensor(v):
                replan_info[k] = v[replan_idx_tensor]
            elif isinstance(v, np.ndarray):
                replan_info[k] = v[replan_idx]
            elif isinstance(v, list):
                replan_info[k] = [v[i] for i in replan_idx]
            else:
                replan_info[k] = v

        outputs = self.solver(replan_info)
        best_actions = outputs["actions"].to(self.device)

        plan = best_actions[:, : self._plan_cfg.receding_horizon]
        decoded_actions = self.jepa.get_executable_actions(replan_info, plan)
        decoded_actions_np = decoded_actions.cpu().numpy()

        if "action" in self.process:
            n_sub, t_phys, s_dim = decoded_actions_np.shape
            flat = decoded_actions_np.reshape(-1, s_dim)
            flat = self.process["action"].inverse_transform(flat)
            decoded_actions_np = flat.reshape(n_sub, t_phys, s_dim)

        self._state.commit_plan(replan_idx, decoded_actions_np)

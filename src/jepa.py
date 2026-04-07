import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from typing import Dict

from src.config import ModelConfig


def detach_clone(v: torch.Tensor) -> torch.Tensor:
    return v.detach().clone() if torch.is_tensor(v) else v


class StandardJEPA(nn.Module):
    """Predicts future states using raw physical actions."""

    def __init__(
        self,
        config: ModelConfig,
        encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        projector: nn.Module = None,
        pred_proj: nn.Module = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")

        pixels_emb = self.encoder(pixels)
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        b = emb.size(0)
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=b)

        return preds

    def rollout(
        self,
        info: Dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        if history_size is None:
            history_size = self.config.dataset.history_size
        H = info["pixels"].size(2) if "pixels" in info else 1
        B, S, T = action_sequence.shape[:3]

        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            emb_trunc = emb[:, -HS:]
            act_trunc = act[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        emb_trunc = emb[:, -HS:]
        act_trunc = act[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(
        self, info_dict: Dict[str, torch.Tensor], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in list(info_dict.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)

    def get_executable_actions(
        self, info_dict: Dict[str, torch.Tensor], best_sequence: torch.Tensor
    ) -> torch.Tensor:
        # For standard JEPA, the sequence IS the physical action sequence.
        # We return the first action as a chunk of length 1.
        return best_sequence[0].unsqueeze(0)


# TODO: Review skill JEPA in both training and planning methods
class SkillJEPA(nn.Module):
    """Predicts future states over a latent action manifold (macro-steps)."""

    def __init__(
        self,
        config: ModelConfig,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module = None,
        pred_proj: nn.Module = None,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")

        pixels_emb = self.encoder(pixels)
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            state_anchors = info["emb"][:, 0]
            latent, recon, mu, logvar = self.action_encoder(
                info["action"], state_anchors
            )
            info["act_emb"] = latent.unsqueeze(1)
            info["recon_actions"] = recon
            info["mu"] = mu
            info["logvar"] = logvar

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(
        self,
        info: Dict[str, torch.Tensor],
        skill_sequence: torch.Tensor,
        history_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        if history_size is None:
            history_size = self.config.dataset.history_size
        H = info["pixels"].size(2) if "pixels" in info else 1
        B, S, M = skill_sequence.shape[:3]

        act_0, act_future = torch.split(skill_sequence, [H, M - H], dim=2)
        n_steps = M - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            emb_trunc = emb[:, -HS:]
            act_trunc = act[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        emb_trunc = emb[:, -HS:]
        act_trunc = act[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(
        self, info_dict: Dict[str, torch.Tensor], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in list(info_dict.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)

    # TODO: Understand these mechanics well
    def get_executable_actions(
        self, info_dict: Dict[str, torch.Tensor], best_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Decodes the first latent macro-step into a physical action chunk."""
        with torch.no_grad():
            encoded = self.encode(info_dict)
            state_emb = encoded["emb"][:, 0]  # Anchor state (1, D_state)

            # Extract first latent
            first_latent = (
                best_sequence[0] if best_sequence.ndim == 2 else best_sequence
            )
            first_latent = torch.tensor(
                first_latent, device=state_emb.device
            ).unsqueeze(0)

            fused = torch.cat((state_emb, first_latent), dim=-1)
            physical_chunk = self.action_encoder.decoder(fused)
            physical_chunk = physical_chunk.view(-1, self.config.action.space_dim)
            return physical_chunk

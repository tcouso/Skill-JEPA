from abc import abstractmethod

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.config import JEPAConfig
from src.modules.observation import InverseModel, SIGReg, VCReg


class BaseJEPA(nn.Module):
    """Shared observation encoding, latent prediction, rollout and cost logic."""

    def __init__(
        self,
        jepa_config: JEPAConfig,
        obs_enc: nn.Module,
        obs_predictor: nn.Module,
        act_projector: nn.Module,
        obs_proj: nn.Module,
        pred_proj: nn.Module,
        proprio_enc: nn.Module | None = None,
    ):
        super().__init__()
        self.config = jepa_config
        self.obs_enc = obs_enc
        self.obs_predictor = obs_predictor
        self.act_projector = act_projector
        self.obs_proj = obs_proj
        self.pred_proj = pred_proj
        self.proprio_enc = proprio_enc
        self.sigreg = SIGReg(
            knots=jepa_config.sigreg.knots, num_proj=jepa_config.sigreg.num_proj
        )
        self.vcreg = VCReg()
        self.inverse_model: nn.Module | None = None
        if jepa_config.regularizer == "smwm":
            self.inverse_model = InverseModel(
                embed_dim=jepa_config.vision.embed_dim,
                action_dim=jepa_config.dataset.frameskip * jepa_config.action.space_dim,
                hidden_dim=jepa_config.smwm.hidden_dim,
            )

    @property
    def plan_dim(self) -> int:
        """Dims CEM searches per macro-step."""
        return self.act_projector.plan_dim

    def encode_obs(
        self, pixels: torch.Tensor, proprio: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(b, t, C, H, W) and (b, t, p) -> (b, t, d); proprio is ignored without a proprio encoder."""
        pixels = pixels.float()
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        obs_emb = self.obs_proj(self.obs_enc(pixels))

        if self.proprio_enc is not None:
            if proprio is None:
                raise ValueError("config.proprio.dim is set but no proprio was passed")
            proprio = proprio[..., self.config.proprio.start :]
            proprio = rearrange(proprio.to(obs_emb.device).float(), "b t p -> (b t) p")
            obs_emb = torch.cat([obs_emb, self.proprio_enc(proprio)], dim=-1)

        return rearrange(obs_emb, "(b t) d -> b t d", b=batch_size)

    def predict_latent_obs(
        self, obs_emb: torch.Tensor, act_cond: torch.Tensor
    ) -> torch.Tensor:
        batch_size = obs_emb.size(0)
        preds = self.obs_predictor(obs_emb, act_cond)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))

        return rearrange(preds, "(b t) d -> b t d", b=batch_size)

    def compute_prediction_loss(
        self, obs_emb: torch.Tensor, act_cond: torch.Tensor
    ) -> torch.Tensor:
        history_size = self.config.dataset.history_size
        pred_horizon = self.config.dataset.pred_horizon

        obs_ctx_emb = obs_emb[:, :history_size]
        act_cond = act_cond[:, :history_size]
        obs_tgt_emb = obs_emb[:, pred_horizon:]

        pred_emb = self.predict_latent_obs(obs_ctx_emb, act_cond)

        return F.mse_loss(pred_emb, obs_tgt_emb)

    def compute_regularization(
        self, obs_emb: torch.Tensor, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Anti-collapse terms on (b, t, d) embeddings; "reg_loss" is their weighted sum."""
        match self.config.regularizer:
            case "sigreg":
                sigreg_loss = self.sigreg(obs_emb.transpose(0, 1))
                return {
                    "reg_loss": self.config.sigreg.weight * sigreg_loss,
                    "sigreg_loss": sigreg_loss,
                }

            case "pldm":
                weights = self.config.pldm
                terms = self.vcreg(obs_emb)
                terms["temp_align_loss"] = F.mse_loss(obs_emb[:, :-1], obs_emb[:, 1:])
                terms["reg_loss"] = (
                    weights.std_weight * terms["std_loss"]
                    + weights.std_t_weight * terms["std_t_loss"]
                    + weights.cov_weight * terms["cov_loss"]
                    + weights.cov_t_weight * terms["cov_t_loss"]
                    + weights.temp_align_weight * terms["temp_align_loss"]
                )
                return terms

            case "smwm":
                pred_action = self.inverse_model(obs_emb[:, :-1], obs_emb[:, 1:])
                inverse_loss = F.mse_loss(pred_action, action[:, :-1])
                return {
                    "reg_loss": self.config.smwm.inverse_weight * inverse_loss,
                    "inverse_loss": inverse_loss,
                }

            case _:
                raise ValueError(f"Unknown regularizer: {self.config.regularizer!r}")

    @abstractmethod
    def encode_act(self, action: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Action chunks -> a fresh dict, always including "act_cond"."""
        ...

    @abstractmethod
    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def get_executable_actions(
        self, info_dict: dict, best_sequence: torch.Tensor
    ) -> torch.Tensor: ...

    def rollout(
        self,
        pixels: torch.Tensor,
        proprio: torch.Tensor | None,
        action_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Roll the predictor forward over one CEM candidate population.

        pixels:          (b, cem_num_samples, num_obs_frames, C, H, W)
        proprio:         (b, cem_num_samples, num_obs_frames, p) or None
        action_sequence: (b, cem_num_samples, planning_horizon, plan_dim)
        returns:         (predicted_emb, num_obs_frames)
        """
        history_size = self.config.dataset.history_size
        num_obs_frames = pixels.size(2)
        batch_size, cem_num_samples, planning_horizon = action_sequence.shape[:3]
        n_future_steps = planning_horizon - num_obs_frames

        # Upstream env_pool.py:177 makes the context axis a singleton.
        act_context, act_future = torch.split(
            action_sequence, [num_obs_frames, n_future_steps], dim=2
        )

        obs_emb = self.encode_obs(pixels[:, 0], None if proprio is None else proprio[:, 0])
        emb = obs_emb.unsqueeze(1).expand(batch_size, cem_num_samples, -1, -1)

        emb = rearrange(emb, "b n ... -> (b n) ...").clone()
        act = rearrange(act_context, "b n ... -> (b n) ...")
        act_future = rearrange(act_future, "b n ... -> (b n) ...")

        for t in range(n_future_steps):
            act_cond = self.act_projector(act)
            emb_trunc = emb[:, -history_size:]
            act_trunc = act_cond[:, -history_size:]
            pred_emb = self.predict_latent_obs(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            act = torch.cat([act, act_future[:, t : t + 1, :]], dim=1)

        act_cond = self.act_projector(act)
        emb_trunc = emb[:, -history_size:]
        act_trunc = act_cond[:, -history_size:]
        pred_emb = self.predict_latent_obs(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        predicted_emb = rearrange(
            emb, "(b n) ... -> b n ...", b=batch_size, n=cem_num_samples
        )

        return predicted_emb, num_obs_frames

    def criterion(
        self,
        predicted_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        num_obs_frames: int,
    ) -> torch.Tensor:
        pred_emb = predicted_emb
        # Without the unsqueeze, expand aligns batch against cem_num_samples and breaks above batch 1.
        goal_emb = goal_emb[..., -1:, :].unsqueeze(1).expand_as(pred_emb)

        match self.config.evaluation.cost_mode:
            case "running":
                goal_cost = (
                    F.mse_loss(
                        pred_emb[..., num_obs_frames:, :],
                        goal_emb[..., num_obs_frames:, :].detach(),
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean(dim=-1)
                )

            case "terminal":
                pred_last_emb = pred_emb[..., -1:, :]
                goal_last_emb = goal_emb[..., -1:, :]

                goal_cost = F.mse_loss(
                    pred_last_emb, goal_last_emb.detach(), reduction="none"
                ).sum(dim=tuple(range(2, pred_emb.ndim)))

        return goal_cost

    def _from_search_space(self, candidates: torch.Tensor) -> torch.Tensor:
        """CEM's coordinates as macro_actions; an affine arms may rescale."""
        return candidates

    def get_cost(
        self, info_dict: dict[str, torch.Tensor], action_candidates: torch.Tensor
    ) -> torch.Tensor:
        action_candidates = self._from_search_space(action_candidates)
        device = next(self.parameters()).device
        pixels = info_dict["pixels"].to(device)
        proprio = info_dict.get("proprio")
        goal_proprio = info_dict.get("goal_proprio")

        # Upstream cem.py:158-165 expands one observation across the population.
        goal_emb = self.encode_obs(
            info_dict["goal"][:, 0].to(device),
            None if goal_proprio is None else goal_proprio[:, 0],
        )

        predicted_emb, num_obs_frames = self.rollout(pixels, proprio, action_candidates)
        return self.criterion(predicted_emb, goal_emb, num_obs_frames)


class RawJEPA(BaseJEPA):
    """Predicts future states using raw physical actions."""

    def encode_act(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"act_cond": self.act_projector(action)}

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        action = torch.nan_to_num(batch["action"], 0.0)

        obs_emb = self.encode_obs(batch["pixels"], batch.get("proprio"))
        act_cond = self.encode_act(action)["act_cond"]

        pred_loss = self.compute_prediction_loss(obs_emb, act_cond)
        reg_terms = self.compute_regularization(obs_emb, action)
        loss = pred_loss + reg_terms["reg_loss"]

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            **reg_terms,
        }

    def get_executable_actions(
        self, info_dict: dict[str, torch.Tensor], best_sequence: torch.Tensor
    ) -> torch.Tensor:
        frameskip = self.config.dataset.frameskip
        return rearrange(best_sequence, "b t (f d) -> b (t f) d", f=frameskip)


class SkillJEPA(BaseJEPA):
    """Predicts future states over a latent action manifold."""

    def __init__(
        self,
        jepa_config: JEPAConfig,
        obs_enc: nn.Module,
        obs_predictor: nn.Module,
        act_projector: nn.Module,
        obs_proj: nn.Module,
        pred_proj: nn.Module,
        skill_encoder: nn.Module,
        skill_bottleneck: nn.Module,
        skill_decoder: nn.Module,
        proprio_enc: nn.Module | None = None,
    ):
        super().__init__(
            jepa_config, obs_enc, obs_predictor, act_projector, obs_proj, pred_proj, proprio_enc
        )
        self.skill_encoder = skill_encoder
        self.skill_bottleneck = skill_bottleneck
        self.skill_decoder = skill_decoder
        self.kld_weight = jepa_config.action.kld_weight
        self.normalize_kld = jepa_config.action.normalize_kld

    def encode_act(
        self, action: torch.Tensor, obs_emb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size = action.size(0)
        frameskip = self.config.dataset.frameskip
        act_chunk = rearrange(action, "b t (f d) -> (b t) f d", f=frameskip)

        obs_cond = self._conditioning_obs(obs_emb).unsqueeze(-2)  # b,t,d -> b,t,1,d
        obs_cond = rearrange(obs_cond, "b t ... -> (b t) ...")

        bottleneck_out = self.skill_bottleneck(self.skill_encoder(act_chunk, obs_cond))
        skill = bottleneck_out["skill"]

        proj_in = skill.detach() if self.config.action.detach_action_conditioning else skill

        return {
            "act_chunk": rearrange(act_chunk, "(b t) ... -> b t ...", b=batch_size),
            "skill": rearrange(skill, "(b t) ... -> b t ...", b=batch_size),
            "skill_mu": rearrange(
                bottleneck_out["skill_mu"], "(b t) ... -> b t ...", b=batch_size
            ),
            "skill_logvar": rearrange(
                bottleneck_out["skill_logvar"], "(b t) ... -> b t ...", b=batch_size
            ),
            "act_cond": rearrange(
                self.act_projector(proj_in), "(b t) ... -> b t ...", b=batch_size
            ),
        }

    def _conditioning_obs(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """The observation used as a state token by the skill encoder/decoder."""
        return obs_emb.detach() if self.config.action.detach_state_conditioning else obs_emb

    def decode_act(self, skill: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        batch_size = skill.size(0)
        obs_cond = self._conditioning_obs(obs_cond).unsqueeze(-2)
        obs_cond = rearrange(obs_cond, "b t ... -> (b t) ...")
        skill = rearrange(skill, "b t ... -> (b t) ...")
        act_pred = self.skill_decoder(skill, obs_cond)

        return rearrange(act_pred, "(b t) ... -> b t ...", b=batch_size)

    def compute_kld_loss(
        self, skill_mu: torch.Tensor, skill_logvar: torch.Tensor
    ) -> torch.Tensor:
        reduce = torch.mean if self.normalize_kld else torch.sum

        return torch.mean(
            -0.5 * reduce(1 + skill_logvar - skill_mu**2 - skill_logvar.exp(), dim=-1)
        )

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        action = torch.nan_to_num(batch["action"], 0.0)

        obs_emb = self.encode_obs(batch["pixels"], batch.get("proprio"))
        act_output = self.encode_act(action, obs_emb)

        pred_loss = self.compute_prediction_loss(obs_emb, act_output["act_cond"])
        reg_terms = self.compute_regularization(obs_emb, action)

        skill, decoder_obs = act_output["skill"], obs_emb
        if self.config.action.detach_action_decoding:
            skill, decoder_obs = skill.detach(), obs_emb.detach()
        pred_act_chunk = self.decode_act(skill, decoder_obs)
        act_recon_loss = F.mse_loss(pred_act_chunk, act_output["act_chunk"])

        kld_loss = torch.zeros((), device=pred_loss.device)
        if self.config.action_encoder == "vae":
            kld_loss = self.compute_kld_loss(
                act_output["skill_mu"], act_output["skill_logvar"]
            )

        loss = (
            pred_loss
            + reg_terms["reg_loss"]
            + act_recon_loss
            + self.kld_weight * kld_loss
        )

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            **reg_terms,
            "act_recon_loss": act_recon_loss,
            "kld_loss": kld_loss,
        }

    def _skill_prior(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.skill_bottleneck
        return (
            bottleneck.skill_prior_mu.to(like.device, like.dtype),
            bottleneck.skill_prior_var.to(like.device, like.dtype),
        )

    def _from_search_space(self, candidates: torch.Tensor) -> torch.Tensor:
        if not self.config.evaluation.whiten_skill_space:
            return candidates
        mu, var = self._skill_prior(candidates)

        return candidates * var.sqrt() + mu

    @torch.no_grad()
    def get_executable_actions(
        self, info_dict: dict[str, torch.Tensor], best_sequence: torch.Tensor
    ) -> torch.Tensor:
        num_macro_actions = best_sequence.size(1)

        best_sequence = self._from_search_space(best_sequence)
        pixels = info_dict["pixels"].unsqueeze(1)
        proprio = info_dict.get("proprio")
        if proprio is not None:
            proprio = proprio.unsqueeze(1)
        action_seq = best_sequence.unsqueeze(1)
        predicted_emb, _ = self.rollout(pixels, proprio, action_seq)

        plan_obs_emb = predicted_emb[:, 0, :num_macro_actions] # (b, num_macro_actions, d)
        decoded = self.decode_act(best_sequence, plan_obs_emb) # (b, t, f, d)

        return rearrange(decoded, "b t f d -> b (t f) d")

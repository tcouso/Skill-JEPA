import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from typing import Dict, Any


def detach_clone(v: torch.Tensor) -> torch.Tensor:
    return v.detach().clone() if torch.is_tensor(v) else v


class StandardJEPA(nn.Module):
    """Predicts future states using raw physical actions."""
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        projector: nn.Module = None,
        pred_proj: nn.Module = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") 
        
        output = self.encoder(pixels)
        pixels_emb = output.last_hidden_state[:, 0] if hasattr(output, "last_hidden_state") else output

        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = info["action"]

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info: Dict[str, torch.Tensor], action_sequence: torch.Tensor, history_size: int = 3) -> Dict[str, torch.Tensor]:
        assert "pixels" in info, "pixels not in info_dict"
        
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
    

class SkillJEPA(nn.Module):
    """Predicts future states over a latent action manifold (macro-steps)."""
    def __init__(
        self,
        config: Any,
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
        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") 
        
        output = self.encoder(pixels)
        pixels_emb = output.last_hidden_state[:, 0] if hasattr(output, "last_hidden_state") else output

        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # Action expected shape: (B, T, K, D_action) where K is sequence length
            state_anchors = info["emb"][:, 0]
            latent, recon, mu, logvar = self.action_encoder(info["action"], state_anchors)
            
            info["act_emb"] = latent.unsqueeze(1) 
            info["recon_actions"] = recon
            info["mu"] = mu
            info["logvar"] = logvar

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info: Dict[str, torch.Tensor], skill_sequence: torch.Tensor, history_size: int = 3) -> Dict[str, torch.Tensor]:
        """
        skill_sequence: (B, S, M, D_latent) where M is the number of macro-steps.
        """
        assert "pixels" in info, "pixels not in info_dict"
        
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
        
        # Rollout in Latent Space (Macro-steps)
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

        # Physical Materialization
        flat_states = rearrange(emb[:, :-1, :], "bs m d -> (bs m) d")
        flat_skills = rearrange(act, "bs m d -> (bs m) d")
        
        fused = torch.cat((flat_states, flat_skills), dim=-1)
        decoded_chunks = self.action_encoder.decoder(fused)
        
        decoded_chunks = decoded_chunks.view(
            B, S, M, self.config.action_sequence_length, self.config.action_space_dim
        )
        info["decoded_physical_actions"] = rearrange(decoded_chunks, "b s m k d -> b s (m k) d")

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
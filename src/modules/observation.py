import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from transformers import ViTConfig, ViTModel


class VisionEncoder(ViTModel):
    _SCALE_CONFIGS = {
        "tiny": dict(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
        ),
        "small": dict(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
        ),
        "base": dict(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
    }

    def __init__(
        self, encoder_scale: str = "tiny", patch_size: int = 14, img_size: int = 224
    ):
        scale_cfg = self._SCALE_CONFIGS[encoder_scale]
        vit_config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            **scale_cfg,
        )
        super().__init__(vit_config)
        self.native_dim: int = scale_cfg["hidden_size"]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        output = super().forward(pixels, interpolate_pos_encoding=True)
        
        return output.last_hidden_state[:, 0]


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer, as in LeWM loss"""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class VCReg(nn.Module):
    """Variance-Covariance Regularizer, as in PLDM loss."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z: torch.Tensor) -> torch.Tensor:
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, "t b i, t b j -> t i j") / (B - 1)  # (T, D, D)
        diag = einsum(cov, "t i i -> t i").pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, T, D) -> per-term losses."""
        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(dim=0, keepdim=True)

        return {
            "std_loss": self._std_loss(z).mean(),
            "std_t_loss": self._std_loss(z.transpose(0, 1)).mean(),
            "cov_loss": self._cov_loss(z).mean(),
            "cov_t_loss": self._cov_loss(z.transpose(0, 1)).mean(),
        }


class InverseModel(nn.Module):
    """Predicts the action chunk from consecutive embeddings, as in the sensorimotor-world-model repo."""

    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp1], dim=-1))

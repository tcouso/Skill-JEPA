"""Action encoder metrics: reconstruction error, and the linear floor it is read against."""

import torch
import torch.nn.functional as F


def floor_mse(chunks: torch.Tensor, rank: int) -> float:
    """Variance the best `rank`-dimensional linear projection of (n, frameskip, space_dim) chunks discards."""
    flat = chunks.reshape(chunks.shape[0], -1).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (flat.shape[0] - 1)
    spectrum = torch.linalg.eigvalsh(covariance).flip(0).clamp(min=0.0)
    return float(1.0 - spectrum[:rank].sum() / spectrum.sum())


def reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor) -> dict:
    per_sample_mse = F.mse_loss(recon, target, reduction="none").mean(dim=(1, 2))
    return {
        "action_recon/mse": per_sample_mse.mean().item(),
        "action_recon/mse_std": per_sample_mse.std().item(),
        "action_recon/n_samples": target.shape[0],
    }

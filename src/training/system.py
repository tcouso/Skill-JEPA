import math

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader

from src.config import JEPAConfig
from src.modules import (
    MLP,
    ActionDecoder,
    ActionEncoder,
    ActionProjector,
    AEBottleneck,
    ARPredictor,
    BaseJEPA,
    RawJEPA,
    SkillJEPA,
    VAEBottleneck,
    VisionEncoder,
)

_BOTTLENECKS = {"vae": VAEBottleneck, "ae": AEBottleneck}

_SKILL_PRIOR_FIT_MAX_BATCHES = 50


class SkillJEPASystem(pl.LightningModule):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config

        embed_dim = config.vision.embed_dim
        frameskip = config.dataset.frameskip
        space_dim = config.action.space_dim
        skill_dim = config.action.skill_dim
        is_latent = config.action.representation == "latent"

        chunk_dim = frameskip * space_dim

        plan_dim = skill_dim if is_latent else chunk_dim

        proprio_emb_dim = config.proprio.emb_dim if config.proprio.dim else 0

        self.obs_enc: VisionEncoder = hydra.utils.instantiate(config.observation_encoder)
        self.obs_proj = MLP(
            input_dim=self.obs_enc.native_dim,
            hidden_dim=config.vision.proj_hidden_dim,
            output_dim=embed_dim - proprio_emb_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )
        self.proprio_enc: nn.Module | None = None
        if config.proprio.dim:
            self.proprio_enc = MLP(
                input_dim=config.proprio.dim,
                hidden_dim=config.proprio.hidden_dim,
                output_dim=proprio_emb_dim,
            )
        self.pred_proj = MLP(
            input_dim=embed_dim,
            hidden_dim=config.vision.proj_hidden_dim,
            output_dim=embed_dim,
            norm_fn=torch.nn.BatchNorm1d,
        )

        self.act_projector = ActionProjector(
            macro_action_dim=plan_dim,
            smoothed_dim=chunk_dim,
            hidden_dim=embed_dim,
        )

        self.skill_encoder: nn.Module | None = None
        self.skill_bottleneck: nn.Module | None = None
        self.skill_decoder: nn.Module | None = None

        if is_latent:
            self.skill_encoder = ActionEncoder(
                space_dim=space_dim,
                hidden_dim=embed_dim,
                obs_dim=embed_dim,
                action_chunk_size=frameskip,
                num_attn_heads=config.action.num_attn_heads,
                num_layers=config.action.num_layers,
                mlp_ratio=config.action.mlp_ratio,
                encoder_state_conditioning=config.action.encoder_state_conditioning,
                pos_embedding=config.action.encoder_pos_embedding,
            )
            self.skill_bottleneck = _BOTTLENECKS[config.action_encoder](
                hidden_dim=embed_dim, skill_dim=skill_dim
            )
            self.skill_decoder = ActionDecoder(
                skill_dim=skill_dim,
                obs_dim=embed_dim,
                hidden_dim=embed_dim,
                action_chunk_size=frameskip,
                space_dim=space_dim,
                num_attn_heads=config.action.num_attn_heads,
                num_layers=config.action.num_layers,
                mlp_ratio=config.action.mlp_ratio,
                decoder_state_conditioning=config.action.decoder_state_conditioning,
            )

        self.predictor: ARPredictor = ARPredictor(
            num_frames=config.dataset.history_size,
            depth=config.predictor.num_layers,
            heads=config.predictor.num_attn_heads,
            mlp_dim=config.predictor.mlp_dim,
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            output_dim=embed_dim,
            dim_head=config.predictor.dim_head,
            dropout=config.predictor.dropout,
            emb_dropout=config.predictor.emb_dropout,
        )

        self.jepa: BaseJEPA
        if is_latent:
            self.jepa = SkillJEPA(
                config,
                self.obs_enc,
                self.predictor,
                self.act_projector,
                self.obs_proj,
                self.pred_proj,
                self.skill_encoder,
                self.skill_bottleneck,
                self.skill_decoder,
                self.proprio_enc,
            )
        else:
            self.jepa = RawJEPA(
                config,
                self.obs_enc,
                self.predictor,
                self.act_projector,
                self.obs_proj,
                self.pred_proj,
                self.proprio_enc,
            )

    def training_step(self, batch) -> torch.Tensor:
        losses = self.jepa.compute_loss(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.training.batch_size,
        )

        return losses["loss"]

    def on_train_epoch_end(self) -> None:
        """Fits the skill prior buffers at the end of the last epoch."""
        # Not on_fit_end: Lightning runs it after the last checkpoint is saved.
        if self.current_epoch != self.trainer.max_epochs - 1:
            return
        if not isinstance(self.jepa, SkillJEPA):
            return

        train_dataset = self.trainer.datamodule.train_dataset
        # Unshuffled, the batch cap would take the first rows on disk.
        sample_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        was_training = self.jepa.training
        self.jepa.eval()
        mu, var = self._fit_skill_prior(sample_loader, _SKILL_PRIOR_FIT_MAX_BATCHES)
        self.jepa.train(was_training)
        self.jepa.skill_bottleneck.set_skill_prior(mu, var)
        self.log_dict(self._skill_prior_summary(mu, var))

    @staticmethod
    def _skill_prior_summary(mu: torch.Tensor, var: torch.Tensor) -> dict:
        """The fitted prior, flattened for logging."""
        summary = {
            "skill_prior/mu_norm": mu.norm().item(),
            "skill_prior/mu_abs_max": mu.abs().max().item(),
            "skill_prior/var_mean": var.mean().item(),
            "skill_prior/var_min": var.min().item(),
            "skill_prior/var_max": var.max().item(),
        }
        for d in range(mu.numel()):
            summary[f"skill_prior/mu_{d}"] = mu[d].item()
            summary[f"skill_prior/var_{d}"] = var[d].item()
        return summary

    @torch.no_grad()
    def _fit_skill_prior(
        self, dataloader: DataLoader, max_batches: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dimension (mean, var) of the skill latent over `dataloader`."""
        skills = []
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            pixels = batch["pixels"].to(self.device)
            action = torch.nan_to_num(batch["action"].to(self.device), 0.0)
            obs_emb = self.jepa.encode_obs(pixels, batch.get("proprio"))
            act_output = self.jepa.encode_act(action, obs_emb)
            skills.append(rearrange(act_output["skill"], "b t d -> (b t) d").cpu())

        z = torch.cat(skills, dim=0)
        return z.mean(dim=0), z.var(dim=0)

    def validation_step(self, batch) -> torch.Tensor:
        losses = self.jepa.compute_loss(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.config.training.batch_size,
        )

        return losses["loss"]

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        effective_bs = self.config.training.batch_size * self.trainer.world_size
        scale = (effective_bs / self.config.training.reference_batch_size) ** self.config.training.lr_scale_exponent
        lr = self.config.training.base_learning_rate * scale
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            betas=self.config.training.betas,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.config.training.warmup_fraction * total_steps))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

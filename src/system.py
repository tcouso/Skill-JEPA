import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.config import ModelConfig
from src.module import (
    VisionEncoder,
    ActionAutoencoder,
    ARPredictor,
    SIGReg
)
from src.jepa import StandardJEPA, SkillJEPA

class ModelSystem(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config

        self.vision_encoder = VisionEncoder(
            model_name=config.vision_model_name, 
            target_hidden_dim=config.obs_encoder_hidden_dim
        )
        
        self.sigreg = SIGReg()
        self.sigreg_weight = 1.0 # Lambda for SIGReg
        self.beta_weight = 0.01  # Beta for VAE KL Divergence

        if config.predictor_mode == "jumpy":
            self.action_encoder = ActionAutoencoder(config)
            
            # Predictor must map from VAE latent dim (64) to Vision dim (768)
            # The context `c` here is the compressed skill `w`
            self.predictor = ARPredictor(
                num_frames=config.action_sequence_length,
                depth=config.decoder_num_layers,
                heads=config.obs_encoder_num_attn_heads,
                mlp_dim=config.obs_encoder_hidden_dim * 4,
                input_dim=config.obs_encoder_hidden_dim,
                hidden_dim=config.obs_encoder_hidden_dim,
            )
            self.jepa = SkillJEPA(
                config=config,
                encoder=self.vision_encoder,
                predictor=self.predictor,
                action_encoder=self.action_encoder
            )
        else:
            self.predictor = ARPredictor(
                num_frames=config.action_sequence_length,
                depth=config.decoder_num_layers,
                heads=config.obs_encoder_num_attn_heads,
                mlp_dim=config.obs_encoder_hidden_dim * 4,
                input_dim=config.obs_encoder_hidden_dim,
                hidden_dim=config.obs_encoder_hidden_dim,
            )
            self.jepa = StandardJEPA(
                encoder=self.vision_encoder,
                predictor=self.predictor
            )

    def _shared_step(self, batch) -> dict:
        """
        Expected batch:
        - 'pixels': (B, T, C, H, W)
        - 'action': (B, T, action_dim) 
        """
        # LeWM replaces NaNs at sequence boundaries
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        # 1. Forward Pass (Encode and Predict)
        output = self.jepa.encode(batch)
        
        emb = output["emb"]  # (B, T, D)
        act_emb = output["act_emb"]

        # 2. Predictive Loss Setup
        ctx_len = 1 # Assuming a history size of 1 for now, adjust as needed
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        
        tgt_emb = emb[:, ctx_len:]
        pred_emb = self.jepa.predict(ctx_emb, ctx_act)

        # 3. Loss Calculations
        pred_loss = F.mse_loss(pred_emb, tgt_emb)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss
        losses = {"pred_loss": pred_loss, "sigreg_loss": sigreg_loss}

        # 4. VAE Losses (if jumpy)
        if self.config.predictor_mode == "jumpy":
            recon_actions = output["recon_actions"]
            mu = output["mu"]
            logvar = output["logvar"]
            
            # Reconstruction Loss
            recon_loss = F.mse_loss(recon_actions, batch["action"])
            
            # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

            vae_loss = recon_loss + self.beta_weight * kld_loss
            total_loss += vae_loss
            
            losses.update({"recon_loss": recon_loss, "kld_loss": kld_loss})

        losses["loss"] = total_loss
        return losses

    def training_step(self, batch) -> torch.Tensor:
        losses = self._shared_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.batch_size
        )
        return losses["loss"]

    def validation_step(self, batch) -> torch.Tensor:
        losses = self._shared_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.config.batch_size
        )
        return losses["loss"]

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.base_learning_rate,
            betas=self.config.betas,
        )

        decay_epochs = self.config.max_epochs - self.config.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=self.config.warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_epochs, eta_min=0.0
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
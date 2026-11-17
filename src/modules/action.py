import torch
import torch.nn as nn

from src.modules.transformers import Block, ConditionalBlock, Transformer


class ActionProjector(nn.Module):
    """Lifts a macro_action, the vector CEM searches, into hidden_dim for the predictor."""

    def __init__(
        self,
        macro_action_dim: int,
        smoothed_dim: int,
        hidden_dim: int,
        mlp_scale: int = 4,
    ):
        super().__init__()
        self._macro_action_dim = macro_action_dim
        self.channel_proj = nn.Linear(macro_action_dim, smoothed_dim)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * hidden_dim, hidden_dim),
        )

    @property
    def plan_dim(self) -> int:
        return self._macro_action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x.float())   # (B, T, smoothed_dim)

        return self.embed(x)


class ActionEncoder(nn.Module):
    def __init__(
        self,
        space_dim: int,
        hidden_dim: int,
        obs_dim: int,
        action_chunk_size: int,
        num_attn_heads: int,
        num_layers: int,
        mlp_ratio: int = 4,
        encoder_state_conditioning: bool = False,
        pos_embedding: bool = True,
    ):
        super().__init__()
        self.encoder_state_conditioning = encoder_state_conditioning

        self.encoder_action_projection = nn.Sequential(
            nn.Linear(space_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = (
            nn.Parameter(torch.randn(1, action_chunk_size, hidden_dim)) if pos_embedding else None
        )

        if encoder_state_conditioning:
            self.encoder_obs_projection = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        self.encoder = Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            depth=num_layers,
            heads=num_attn_heads,
            dim_head=hidden_dim // num_attn_heads,
            mlp_dim=hidden_dim * mlp_ratio,
            block_class=Block,
            causal=False,
        )

    def forward(self, actions: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        b = actions.size(0)
        cls = self.cls_token.expand(b, -1, -1)                       # (B, 1, hidden_dim)
        projected_actions = self.encoder_action_projection(actions)  # (B, K, hidden_dim)
        if self.pos_embedding is not None:
            projected_actions = projected_actions + self.pos_embedding

        tokens = [cls]
        if self.encoder_state_conditioning:
            tokens.append(self.encoder_obs_projection(obs_cond))     # (B, 1, hidden_dim)
        tokens.append(projected_actions)

        encoded_sequence = self.encoder(torch.cat(tokens, dim=1))

        return encoded_sequence[:, 0, :]


class ActionDecoder(nn.Module):
    def __init__(
        self,
        skill_dim: int,
        obs_dim: int,
        hidden_dim: int,
        action_chunk_size: int,
        space_dim: int,
        num_attn_heads: int,
        num_layers: int,
        mlp_ratio: int = 4,
        decoder_state_conditioning: bool = True,
    ):
        super().__init__()
        self.decoder_state_conditioning = decoder_state_conditioning
        self.action_chunk_size = action_chunk_size

        self.cond_proj = nn.Linear(skill_dim, hidden_dim)

        if decoder_state_conditioning:
            self.obs_projection = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        self.query_tokens = nn.Parameter(torch.zeros(1, action_chunk_size, hidden_dim))

        dim_head = hidden_dim // num_attn_heads
        self.blocks = nn.ModuleList([
            ConditionalBlock(hidden_dim, num_attn_heads, dim_head, hidden_dim * mlp_ratio, causal=False)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, space_dim)

    def forward(self, latent: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        b = latent.size(0)
        c = self.cond_proj(latent).unsqueeze(1)  # (B, 1, hidden_dim) — broadcasts over sequence

        query = self.query_tokens.expand(b, -1, -1)  # (B, K, hidden_dim)

        if self.decoder_state_conditioning:
            state_token = self.obs_projection(obs_cond)  # (B, 1, hidden_dim)
            x = torch.cat([state_token, query], dim=1)   # (B, 1+K, hidden_dim)
        else:
            x = query

        for block in self.blocks:
            x = block(x, c)

        return self.output_proj(self.norm(x[:, -self.action_chunk_size:]))  # (B, K, space_dim)


class _Bottleneck(nn.Module):
    """Skill latent bottleneck, holding its fitted per-dimension Gaussian prior as buffers."""

    def __init__(self, skill_dim: int):
        super().__init__()
        self.skill_dim = skill_dim
        self.register_buffer("skill_prior_mu", torch.zeros(skill_dim))
        self.register_buffer("skill_prior_var", torch.ones(skill_dim))

    @torch.no_grad()
    def set_skill_prior(self, mu: torch.Tensor, var: torch.Tensor) -> None:
        """Write a fitted prior in, matching the buffers' device and dtype."""
        self.skill_prior_mu.copy_(mu.to(self.skill_prior_mu))
        self.skill_prior_var.copy_(var.to(self.skill_prior_var))


class AEBottleneck(_Bottleneck):
    def __init__(self, hidden_dim: int, skill_dim: int):
        super().__init__(skill_dim)
        self.fc_latent = nn.Linear(hidden_dim, skill_dim)

    def forward(self, encoded_chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.fc_latent(encoded_chunk)

        return {"skill": z, "skill_mu": torch.zeros_like(z), "skill_logvar": torch.zeros_like(z)}


class VAEBottleneck(_Bottleneck):
    def __init__(self, hidden_dim: int, skill_dim: int):
        super().__init__(skill_dim)
        self.fc_mu = nn.Linear(hidden_dim, skill_dim)
        self.fc_logvar = nn.Linear(hidden_dim, skill_dim)

    def forward(self, encoded_chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.fc_mu(encoded_chunk)
        logvar = self.fc_logvar(encoded_chunk).clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        return {"skill": z, "skill_mu": mu, "skill_logvar": logvar}

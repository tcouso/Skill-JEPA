import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.models as tv_models
from typing import Tuple, Optional, Type

from src.config import ModelConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
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


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: Type[nn.Module] = Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
            
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(self, input_dim: int = 10, smoothed_dim: int = 10, emb_dim: int = 10, mlp_scale: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        norm_fn: Type[nn.Module] = nn.LayerNorm,
        act_fn: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        norm_layer = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    def __init__(
        self,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_b_16", target_hidden_dim: int = 768):
        super().__init__()
        
        # Load any torchvision model dynamically with default weights
        self.vit = tv_models.get_model(model_name, weights="DEFAULT")
        
        # Strip the classification head (handles both ViT and ResNet paradigms)
        if hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, "fc"):
            self.vit.fc = nn.Identity()
            
        # Dynamically calculate the backbone's native hidden dimension
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, 3, 224, 224)
            native_dim = self.vit(dummy_tensor).shape[-1]
            
        self.proj = nn.Linear(native_dim, target_hidden_dim) if native_dim != target_hidden_dim else nn.Identity()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.vit(pixels)
        return self.proj(features)


class ActionAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.action_sequence_length = config.action_sequence_length
        self.action_space_dim = config.action_space_dim
        
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_space_dim, config.action_encoder_hidden_dim),
            nn.LayerNorm(config.action_encoder_hidden_dim),
            nn.GELU(),
        )
        
        self.state_projection = nn.Sequential(
            nn.Linear(config.obs_encoder_hidden_dim, config.action_encoder_hidden_dim),
            nn.LayerNorm(config.action_encoder_hidden_dim),
            nn.GELU()        
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.action_encoder_hidden_dim,
            nhead=config.action_encoder_num_attn_heads,
            dim_feedforward=config.action_encoder_hidden_dim * config.action_encoder_mlp_ratio,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.action_encoder_num_layers)
        
        self.fc_mu = nn.Linear(config.action_encoder_hidden_dim, config.action_encoder_hidden_dim)
        self.fc_logvar = nn.Linear(config.action_encoder_hidden_dim, config.action_encoder_hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(config.action_encoder_hidden_dim + config.obs_encoder_hidden_dim, config.obs_encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.obs_encoder_hidden_dim, config.action_sequence_length * config.action_space_dim),
        )

    def forward(self, actions: torch.Tensor, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        projected_actions = self.action_projection(actions)
        projected_states = self.state_projection(state_embedding)
        
        sequence = torch.cat((projected_states.unsqueeze(1), projected_actions), dim=1)
        sequence = self.transformer(sequence)

        cls = sequence[:, 0, :]
        mu = self.fc_mu(cls)
        logvar = self.fc_logvar(cls)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        state_and_latent = torch.cat((state_embedding, latent), dim=1)
        reconstructed_actions = self.decoder(state_and_latent)
        reconstructed_actions = reconstructed_actions.view(-1, self.action_sequence_length, self.action_space_dim)

        return latent, reconstructed_actions, mu, logvar
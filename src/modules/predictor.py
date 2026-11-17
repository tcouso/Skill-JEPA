import torch
import torch.nn as nn

from src.modules.transformers import ConditionalBlock, Transformer


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] | None = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
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
        output_dim: int | None = None,
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

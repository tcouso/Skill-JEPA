import math
import torch
import torch.nn as nn
from typing import Tuple

from src.config import ActSiamMAEConfig


def generate_pos_embeddings(
    hidden_dim: int, grid_side_length: int, seq_length: int
) -> torch.Tensor:
    grid_arange = torch.arange(grid_side_length)
    grid_x, grid_y = torch.meshgrid(grid_arange, grid_arange, indexing="xy")

    omega_i = torch.exp(
        ((-2 * torch.arange(hidden_dim // 4)) / hidden_dim) * math.log(10_000)
    )
    sin_grid_x = torch.sin(grid_x.unsqueeze(-1) * omega_i)
    cos_grid_x = torch.cos(grid_x.unsqueeze(-1) * omega_i)
    grid_x_pos_embeddings = torch.concat((sin_grid_x, cos_grid_x), dim=-1)

    sin_grid_y = torch.sin(grid_y.unsqueeze(-1) * omega_i)
    cos_grid_y = torch.cos(grid_y.unsqueeze(-1) * omega_i)
    grid_y_pos_embeddings = torch.concat((sin_grid_y, cos_grid_y), dim=-1)

    pos_embeddings = torch.concat(
        (grid_x_pos_embeddings, grid_y_pos_embeddings), dim=-1
    )
    pos_embeddings = pos_embeddings.view(seq_length, -1).unsqueeze(0)

    return pos_embeddings


class ActSiamMAEPatchifier(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEPatchifier, self).__init__()
        self.config = config
        self.grid_side_length = config.grid_side_length
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim
        self.seq_length = config.seq_length

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.view(
            -1,
            self.num_channels,
            self.grid_side_length,
            self.patch_size,
            self.grid_side_length,
            self.patch_size,
        )
        frame = frame.permute(0, 2, 4, 1, 3, 5)
        frame = frame.reshape(
            -1, self.seq_length, self.num_channels * self.patch_size * self.patch_size
        )

        return frame


class ActSiamMAEDepatchifier(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEDepatchifier, self).__init__()
        self.config = config
        self.grid_side_length = config.grid_side_length
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim
        self.img_size = config.frame_size
        self.seq_length = config.seq_length

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        patched_frame = embeddings.view(
            -1,
            self.grid_side_length,
            self.grid_side_length,
            self.num_channels,
            self.patch_size,
            self.patch_size,
        )
        patched_frame = patched_frame.permute(0, 3, 1, 4, 2, 5)
        frame = patched_frame.reshape(
            -1, self.num_channels, self.img_size, self.img_size
        )

        return frame


class ActSiamMAEMultiHeadAttention(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEMultiHeadAttention, self).__init__()
        self.config = config
        self.num_attn_heads = config.num_attn_heads
        self.hidden_dim = config.hidden_dim
        self.seq_length = config.seq_length
        self.patch_size = config.patch_size
        self.grid_side_length = config.grid_side_length
        self.num_channels = config.num_channels
        self.head_dimension = config.head_dimension

        self.key_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.query_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_layer = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self, query_embedding: torch.Tensor, key_value_embedding: torch.Tensor
    ) -> torch.Tensor:

        seq_length = query_embedding.shape[1]

        query = self.query_layer(query_embedding)
        key = self.key_layer(key_value_embedding)
        value = self.value_layer(key_value_embedding)

        # Reshape for multi-headed attention
        query = query.view(-1, seq_length, self.num_attn_heads, self.head_dimension)
        query = query.permute(0, 2, 1, 3)

        key = key.view(-1, seq_length, self.num_attn_heads, self.head_dimension)
        key = key.permute(0, 2, 1, 3)

        value = value.view(-1, seq_length, self.num_attn_heads, self.head_dimension)
        value = value.permute(0, 2, 1, 3)

        simmilarity_scores = torch.matmul(query, torch.transpose(key, 2, 3))
        scaled_simmilarity_scores = simmilarity_scores / math.sqrt(self.head_dimension)

        similarity_probs = torch.softmax(scaled_simmilarity_scores, dim=-1)
        attn = torch.matmul(similarity_probs, value)

        # Restore batch, seq length, embedding shape
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(-1, seq_length, self.hidden_dim)

        return attn


class ActSiamMAEEncoderBlock(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEEncoderBlock, self).__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
        self.multi_head_attn = ActSiamMAEMultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        norm_embeddings = self.layer_norm1(embeddings)
        attn_embeddings = embeddings + self.multi_head_attn(
            query_embedding=norm_embeddings, 
            key_value_embedding=norm_embeddings
        )
        mlp_embeddings = attn_embeddings + self.mlp(self.layer_norm2(attn_embeddings))

        return mlp_embeddings


# TODO: We are lacking a [CLS] token. This is important for linear probing of the model
class ActSiamMAEEncoder(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEEncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.seq_length = config.seq_length
        self.hidden_dim = config.hidden_dim
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.masking_ratio = config.masking_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

        pos_embeddings = generate_pos_embeddings(
            hidden_dim=config.hidden_dim,
            grid_side_length=config.grid_side_length,
            seq_length=config.seq_length,
        )
        cls_pos_embedding = torch.zeros(1, 1, config.hidden_dim)
        full_pos_embedding = torch.cat((cls_pos_embedding, pos_embeddings), dim=1)
        self.register_buffer("pos_embeddings", full_pos_embedding)

        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.patch_layer = ActSiamMAEPatchifier(config)
        self.embed_layer = nn.Linear(
            in_features=config.num_channels * config.patch_size * config.patch_size,
            out_features=config.hidden_dim,
        )
        self.attn_blocks = nn.ModuleList(
            [ActSiamMAEEncoderBlock(config) for _ in range(config.encoder_num_layers)]
        )

    def forward(
        self, past_frame: torch.Tensor, future_frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = past_frame.shape[0]

        # Past frame is given complete
        past_frame = self.patch_layer(past_frame)
        past_embeddings = self.embed_layer(past_frame)
        cls_tokens_past = self.cls_token.expand(batch_size, -1, -1)

        # Prepend [CLS] token for ending probing
        past_embeddings = torch.cat((cls_tokens_past, past_embeddings), dim=1)
        past_embeddings += self.pos_embeddings

        for attn_block in self.attn_blocks:
            past_embeddings = attn_block(past_embeddings)

        past_embeddings = self.layer_norm(past_embeddings)

        # Future frame is masked
        future_frame = self.patch_layer(future_frame)
        future_embeddings = self.embed_layer(future_frame)

        # Omit [CLS] positional embedding for future frame
        future_embeddings += self.pos_embeddings[:, 1:, :]

        num_keep = int(self.seq_length * (1 - self.masking_ratio))
        rand_tensor = torch.rand(batch_size, self.seq_length, device=self.device)
        _, ids_shuffle = rand_tensor.sort(dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_shuffle_expanded = ids_shuffle.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        future_embeddings = torch.gather(
            future_embeddings, dim=1, index=ids_shuffle_expanded
        )
        future_embeddings = future_embeddings[:, :num_keep]

        ids_keep = ids_shuffle[:, :num_keep]
        mask = torch.ones(batch_size, self.seq_length, device=self.device)
        mask.scatter_(1, ids_keep, 0)

        # [CLS] token should never be masked, so we add it afterwards
        cls_pos = self.pos_embeddings[:, :1, :]
        cls_tokens_future = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens_future = cls_tokens_future + cls_pos
        future_embeddings = torch.cat((cls_tokens_future, future_embeddings), dim=1)

        for attn_block in self.attn_blocks:
            future_embeddings = attn_block(future_embeddings)

        future_embeddings = self.layer_norm(future_embeddings)

        past_cls = past_embeddings[:, 0, :]
        future_cls = future_embeddings[:, 0, :]

        past_embeddings_without_cls = past_embeddings[:, 1:, :]
        future_embeddings_without_cls = future_embeddings[:, 1:, :]


        return past_embeddings_without_cls, future_embeddings_without_cls, past_cls, future_cls, mask, ids_restore


class ActSiamMAEDecoderBlock(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEDecoderBlock, self).__init__()
        self.config = config
        self.num_attn_heads = config.num_attn_heads
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
        self.layer_norm3 = nn.LayerNorm(config.hidden_dim)
        self.multi_head_self_attn = ActSiamMAEMultiHeadAttention(config)
        self.multi_head_cross_attn = ActSiamMAEMultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self, past_embeddings: torch.Tensor, future_embeddings: torch.Tensor
    ) -> torch.Tensor:
        norm_future_embeddings = self.layer_norm1(future_embeddings)
        attn_embeddings = future_embeddings + self.multi_head_cross_attn(
            query_embedding=norm_future_embeddings, 
            key_value_embedding=past_embeddings
        )
        
        norm_attn_embeddings = self.layer_norm2(attn_embeddings)
        attn_embeddings = attn_embeddings + self.multi_head_self_attn(
            query_embedding=norm_attn_embeddings, 
            key_value_embedding=norm_attn_embeddings
        )
        
        mlp_embeddings = attn_embeddings + self.mlp(self.layer_norm3(attn_embeddings))

        return mlp_embeddings


class ActSiamMAEDecoder(nn.Module):
    def __init__(self, config: ActSiamMAEConfig):
        super(ActSiamMAEDecoder, self).__init__()
        self.config = config
        self.seq_length = config.seq_length
        self.hidden_dim = config.hidden_dim
        pos_embeddings = generate_pos_embeddings(
            hidden_dim=config.hidden_dim,
            grid_side_length=config.grid_side_length,
            seq_length=config.seq_length,
        )
        self.register_buffer("pos_embeddings", pos_embeddings)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.output_layer = nn.Linear(
            config.hidden_dim,
            config.num_channels * config.patch_size * config.patch_size,
        )
        self.attn_blocks = nn.ModuleList(
            [ActSiamMAEDecoderBlock(config) for _ in range(config.decoder_num_layers)]
        )

    def forward(
        self,
        past_embeddings: torch.Tensor,
        future_embeddings: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = future_embeddings.shape[0]
        masked_seq_length = future_embeddings.shape[1]

        mask_embeddings = self.mask_token.repeat(
            batch_size, self.seq_length - masked_seq_length, 1
        )
        future_embeddings = torch.concat((future_embeddings, mask_embeddings), dim=1)
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        future_embeddings = torch.gather(
            future_embeddings, dim=1, index=ids_restore_expanded
        )
        future_embeddings += self.pos_embeddings

        for attn_block in self.attn_blocks:
            future_embeddings = attn_block(past_embeddings, future_embeddings)

        future_embeddings = self.layer_norm(future_embeddings)
        patches = self.output_layer(future_embeddings)

        return patches

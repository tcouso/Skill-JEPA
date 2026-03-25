import math
import torch
import torch.nn as nn
from typing import Tuple

from src.config import ModelConfig


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


class Patchifier(nn.Module):
    def __init__(self, grid_side_length: int, num_channels: int, patch_size: int, seq_length:int):
        super(Patchifier, self).__init__()
        self.grid_side_length = grid_side_length
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.seq_length = seq_length

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


class Depatchifier(nn.Module):
    def __init__(self, grid_side_length: int, num_channels: int, patch_size: int, frame_size: int, obs_encoder_hidden_dim: int, obs_encoder_seq_length: int):
        super(Depatchifier, self).__init__()
        self.grid_side_length = grid_side_length
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_dim = obs_encoder_hidden_dim
        self.img_size = frame_size
        self.seq_length = obs_encoder_seq_length

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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attn_heads: int, hidden_dim: int, head_dimension: int):
        super(MultiHeadAttention, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.hidden_dim = hidden_dim
        self.head_dimension = head_dimension

        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

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

        # Attention computation
        simmilarity_scores = torch.matmul(query, torch.transpose(key, 2, 3))
        scaled_simmilarity_scores = simmilarity_scores / math.sqrt(self.head_dimension)

        similarity_probs = torch.softmax(scaled_simmilarity_scores, dim=-1)
        attn = torch.matmul(similarity_probs, value)

        # Restore batch, seq length, embedding shape
        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(-1, seq_length, self.hidden_dim)

        return attn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_attn_heads: int, head_dimension: int, obs_encoder_mlp_ratio: int):
        super(TransformerEncoderBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.multi_head_attn = MultiHeadAttention(
            num_attn_heads=num_attn_heads,
            hidden_dim=hidden_dim,
            head_dimension=head_dimension,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, obs_encoder_mlp_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(obs_encoder_mlp_ratio * hidden_dim, hidden_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        norm_embeddings = self.layer_norm1(embeddings)
        attn_embeddings = embeddings + self.multi_head_attn(
            query_embedding=norm_embeddings, 
            key_value_embedding=norm_embeddings
        )
        mlp_embeddings = attn_embeddings + self.mlp(self.layer_norm2(attn_embeddings))

        return mlp_embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.seq_length = config.obs_encoder_seq_length
        self.hidden_dim = config.obs_encoder_hidden_dim
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.masking_ratio = config.start_masking_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.obs_encoder_hidden_dim))

        self.masking_ratio = config.start_masking_ratio
        pos_embeddings = generate_pos_embeddings(
            hidden_dim=config.obs_encoder_hidden_dim,
            grid_side_length=config.grid_side_length,
            seq_length=config.obs_encoder_seq_length,
        )
        cls_pos_embedding = torch.zeros(1, 1, config.obs_encoder_hidden_dim)
        full_pos_embedding = torch.cat((cls_pos_embedding, pos_embeddings), dim=1)
        self.register_buffer("pos_embeddings", full_pos_embedding)

        self.layer_norm = nn.LayerNorm(config.obs_encoder_hidden_dim)
        self.patch_layer = Patchifier(
            grid_side_length=config.grid_side_length,
            num_channels=config.num_channels,
            patch_size=config.patch_size,
            seq_length=config.obs_encoder_seq_length,
        )
        self.embed_layer = nn.Linear(
            in_features=config.num_channels * config.patch_size * config.patch_size,
            out_features=config.obs_encoder_hidden_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(
                hidden_dim=config.action_encoder_hidden_dim,
                num_attn_heads=config.obs_encoder_num_attn_heads,
                head_dimension=config.obs_encoder_head_dimension,
                obs_encoder_mlp_ratio=config.obs_encoder_mlp_ratio,
            ) for _ in range(config.action_encoder_num_layers)]
        )


    def forward(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = frame.shape[0]
        frame = self.patch_layer(frame)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = self.embed_layer(frame)
        sequence = torch.cat((cls_tokens, sequence), dim=1)
        sequence += self.pos_embeddings

        for block in self.transformer_blocks:
            sequence = block(sequence)

        sequence = self.layer_norm(sequence)

        cls = sequence[:, 0, :]
        sequence_without_cls = sequence[:, 1:, :]

        return sequence_without_cls, cls


class ActionAutoencoder(nn.Module):

    def __init__(self, config: ModelConfig):
        super(ActionAutoencoder, self).__init__()
        self.config = config
        self.action_sequence_length = config.action_sequence_length
        self.action_space_dim = config.action_space_dim
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_space_dim, config.action_encoder_hidden_dim),
            nn.LayerNorm(config.obs_encoder_hidden_dim),
            nn.GELU(),
            )
        self.state_projection = nn.Sequential(
            nn.Linear(config.obs_encoder_hidden_dim, config.action_encoder_hidden_dim),
            nn.LayerNorm(config.action_encoder_hidden_dim),
            nn.GELU()        
        )
        
        self.transformer_blocks = nn.ModuleList(
                    [TransformerEncoderBlock(
                        hidden_dim=config.action_encoder_hidden_dim,
                        num_attn_heads=config.action_encoder_num_attn_heads,
                        head_dimension=config.obs_encoder_head_dimension,
                        obs_encoder_mlp_ratio=config.action_encoder_mlp_ratio,
                    ) for _ in range(config.action_encoder_num_layers)]
                )
        self.fc_mu = nn.Linear(config.obs_encoder_hidden_dim, config.obs_encoder_hidden_dim)
        self.fc_logvar = nn.Linear(config.obs_encoder_hidden_dim, config.obs_encoder_hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.action_encoder_hidden_dim + config.obs_encoder_hidden_dim, config.obs_encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.obs_encoder_hidden_dim, config.action_sequence_length * config.action_space_dim),
        )

    def forward(self, actions: torch.Tensor, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:        

        projected_actions = self.action_projection(actions)
        projected_states = self.state_projection(state_embedding)
        sequence = torch.concat((projected_states.unsqueeze(1), projected_actions), dim=1)

        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence)

        cls = sequence[:, 0, :]
        mu = self.fc_mu(cls)
        logvar = self.fc_logvar(cls)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std

        state_and_latent = torch.concat((state_embedding, latent), dim=1)
        reconstructed_actions = self.decoder(state_and_latent)
        reconstructed_actions = reconstructed_actions.view(-1, self.action_sequence_length, self.action_space_dim)

        return latent, reconstructed_actions

# TODO: This needs to receive encoded observations, encoded actions (skills), and predict the next encoded observation
# We also need to consider the regularized LeWorldModel loss (with gaussian distribution)
# Finally, we should suppport the baseline, standard version of LeWorldModel

class Predictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Predictor, self).__init__()
        self.config = config
        ...
        

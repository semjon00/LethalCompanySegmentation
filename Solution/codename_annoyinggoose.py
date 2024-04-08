# Swin Transformer based model that leverages a pretrained segmentation model.
# Does not work yet.

from typing import List

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor as T


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(num_patches, num_patches, embed_dim))
        self.pos_embed.normal_(mean=0.0, std=0.02).clamp_(-0.05, 0.05)

    def forward(self, x: T) -> T:
        return x + self.pos_embed


class WindowSelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            window_size: int,  # Swin original
            shift_size: int,  # Swin original
            num_patches: int,  # Swin original
            attn_drop: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (self.embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_patches = num_patches
        if shift_size != 0:
            self.register_buffer('mask', self.calculate_attention_mask())

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: T) -> T:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def window_partition(self, x: T) -> T:
        """
        input shape: batch x height x width x channels
        output shape: number of windows in the batch x flattened window x channels
        """
        return einops.rearrange(x, 'b (h wsa) (w wsb) fp -> (b h w) (wsa wsb) fp',
                                wsa=self.window_size, wsb=self.window_size)

    def window_merging(self, x: T) -> T:
        """
        input shape: number of windows in the batch x flattened window x channels
        output shape: batch x height x width x channels
        """
        # Inverse of window_partition
        num_windows = self.num_patches // self.window_size
        return einops.rearrange(x, '(b h w) (wsa wsb) fp -> b (h wsa) (w wsb) fp',
                                h=num_windows, w=num_windows, wsa=self.window_size, wsb=self.window_size)

    def shift_image(self, x: T, direction: int = -1) -> T:
        """
        input shape: batch x height x width x channels
        output shape: batch x height x width x channels

        direction shows whether the shift is forwards or backwards,
        it can be either 1 or -1
        """
        shift = direction * self.shift_size
        return torch.roll(x, shifts=(shift, shift), dims=(1, 2))

    def calculate_attention_mask(self) -> T:
        """
        output shape: number of windows x flattened window x flattened window
        """
        patch_groups = torch.zeros(self.num_patches, self.num_patches)
        patch_groups[:, -self.shift_size:] = 1
        patch_groups[-self.shift_size:, :] = 2
        patch_groups[-self.shift_size:, -self.shift_size:] = 3

        patch_group_windows = einops.rearrange(
            patch_groups, '(h wsa) (w wsb) -> (h w) (wsa wsb)', wsa=self.window_size, wsb=self.window_size)
        row_vectors = einops.rearrange(patch_group_windows, 'w (n v) -> w n v', n=1)
        column_vectors = einops.rearrange(patch_group_windows, 'w (n v) -> w v n', n=1)
        mask_maps = column_vectors - row_vectors

        mask_binary = mask_maps.clone()
        mask_binary[mask_maps != 0] = 0
        mask_binary[mask_maps == 0] = 1

        return mask_binary

    def forward(self, x: T):
        x = self.shift_image(x, direction=-1)
        x = self.window_partition(x)

        q, k, v = self.project_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale
        if self.shift_size != 0:
            # repeat the mask over batches and heads
            # desired shape: number of windows in the batch x number of heads x flattened window x flattened window
            batch_size = attn_scores.shape[0] // self.mask.shape[0]
            mask = self.mask.unsqueeze(1).repeat(batch_size, self.num_heads, 1, 1)
            attn_scores = attn_scores.masked_fill(mask == 0, -10 ** 9)
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)

        # we merge the windows back into an image
        out = self.window_merging(out)
        # we shift the image back (bottom-right)
        out = self.shift_image(out, direction=1)
        return out


class MLP(nn.Sequential):
    def __init__(self, embed_dim: int, mlp_dim: int):
        super().__init__(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )


class SwinBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        window_size: int,
        shift_size: int,
        num_patches: int,
        attn_drop: float = 0.1,
        drop: float = 0.1,
    ):
        super().__init__()
        self.attn = WindowSelfAttention(
            embed_dim, num_heads, window_size, shift_size, num_patches, attn_drop
        )
        self.ffn = MLP(embed_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x: T) -> T:
        res = x
        out = self.attn(self.norm1(x))
        out = out + self.dropout1(res)
        res = out
        out = self.ffn(self.norm2(out))
        out = out + self.dropout2(res)
        return out


class SwinStage(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        window_size: int,
        num_patches: int,
        num_blocks: int,
        merge_patches: bool = False,
        attn_drop: float = 0.1,
        drop: float = 0.1,
    ):
        super().__init__()
        self.merge_patches = merge_patches
        if self.merge_patches:
            self.patch_merging = nn.Sequential(
                Rearrange('b (h dh) (w wh) c -> b h w (c dh wh)', dh=2, wh=2),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, hidden_dim, window_size,
                      # we will shift by half of the window size every second block
                      window_size // 2 if i % 2 == 1 else 0,
                      num_patches, attn_drop, drop)
            for i in range(num_blocks)
        ])

    def forward(self, x: T) -> T:
        out = self.patch_merging(x) if self.merge_patches else x
        for block in self.blocks:
            out = block(out)
        return out


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        window_size: int,
        in_channels: int,
        embed_dim: int,  # C
        num_heads: int,
        hidden_dim: int,
        num_classes: int,
        stage_sizes: List[int],
        attn_drop: float,
        drop: float
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_projection = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b h w (ph pw c)', ph=patch_size, pw=patch_size),
            nn.Linear(in_channels * patch_size ** 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        num_patches = image_size // patch_size
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        self.dropout = nn.Dropout(drop)
        self.stages = nn.ModuleList([
            SwinStage(
                # the number of patches will decrease by the factor of 2 every stage
                embed_dim, num_heads, hidden_dim, window_size, num_patches // (2 ** i),
                # we will merge patches in every group except for the first one
                num_blocks, merge_patches=i != 0, attn_drop=attn_drop, drop=drop
            ) for i, num_blocks in enumerate(stage_sizes)
        ])
        self.classify = nn.Linear(embed_dim, num_classes)

    def forward(self, x: T):
        out = self.patch_projection(x)
        out = self.pos_embed(out)
        out = self.dropout(out)
        for stage in self.stages:
            out = stage(out)  # Swin stages (linear embedding and swin block)
        out = einops.reduce(out, 'b h w c -> b c', 'mean')
        out = self.classify(out)
        return out

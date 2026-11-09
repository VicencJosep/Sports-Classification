"""VideoMAEv2 ViT-Giant (OpenGVLab) como extractor de rasgos por clip, tal y como lo usa SoccerHigh:
16 fotogramas de 224x224 -> vector de 1408 (media de los tokens + ``fc_norm``).

Código reducido de ``modeling_videomaev2.py`` de ``OpenGVLab/VideoMAEv2-giant`` (BEiT/timm/DeiT); la atención usa
``scaled_dot_product_attention`` (misma matemática, menos memoria). Pesos: ``vit_g_hybrid_pt_1200e_k710_ft.pth`` de
``OpenGVLab/VideoMAE2`` (afinado en Kinetics-710, fp16, 2 GB), en ``models/soccerhigh``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, norm_layer):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, num_frames: int, tubelet_size: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2 * (num_frames // tubelet_size)
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    pos = np.arange(n_position)[:, None]
    i = np.arange(d_hid)[None, :]
    table = pos / np.power(10000, 2 * (i // 2) / d_hid)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float).unsqueeze(0)


class VisionTransformer(nn.Module):
    """``forward`` devuelve el rasgo (B, embed_dim): media de los tokens y ``fc_norm`` (sin la cabeza de clases)."""

    def __init__(self, img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_hidden=6144,
                 num_frames=16, tubelet_size=2, layer_norm_eps=1e-6):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim, num_frames, tubelet_size)
        self.register_buffer("pos_embed", sinusoid_table(self.patch_embed.num_patches, embed_dim), persistent=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_hidden, norm_layer) for _ in range(depth)])
        self.fc_norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc_norm(x.mean(1))


def load_giant_k710(path: Path | str, device: str = "cuda", dtype: torch.dtype = torch.float16) -> VisionTransformer:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("module", ck.get("model", ck))
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    model = VisionTransformer()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected]
    if missing or unexpected:
        raise RuntimeError(f"VideoMAEv2: claves que faltan {missing[:5]}, sobrantes {unexpected[:5]}")
    return model.to(device=device, dtype=dtype).eval()


def normalize_clip(frames_uint8: np.ndarray, device: str, dtype: torch.dtype) -> torch.Tensor:
    """(B, T, H, W, 3) uint8 -> (B, 3, T, H, W) normalizado con media/desviación de ImageNet."""
    x = torch.from_numpy(frames_uint8).to(device=device, non_blocking=True).permute(0, 4, 1, 2, 3).to(dtype) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1, 1)
    return (x - mean) / std

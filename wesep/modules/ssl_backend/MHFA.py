"""MHFA backend for SSL models (copied for wesep).

From the paper:
    An attention-based backend allowing efficient fine-tuning
    of transformer models for speaker verification.

Original authors:
    Junyi Peng, Oldrich Plchot, Themos Stafylakis, Ladislav Mosner,
    Lukas Burget, Jan Cernocky.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore[override]
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad * ctx.scale, None


class SSL_BACKEND_MHFA(nn.Module):
    """Multi-head feature aggregation backend operating on SSL layer outputs.

    Input shape: [B, D, T, L] where:
        B: batch size
        D: feature dimension
        T: frame length
        L: number of transformer layers.
    """

    def __init__(
        self,
        head_nb: int = 8,
        feat_dim: int = 768,
        compression_dim: int = 128,
        embed_dim: int = 256,
        nb_layer: int = 13,
        feature_grad_mult: float = 1.0,
    ) -> None:
        super().__init__()

        self.feature_grad_mult = feature_grad_mult

        # Learnable weights across layers for key/value
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

        self.head_nb = head_nb
        self.ins_dim = feat_dim
        self.cmp_dim = compression_dim
        self.ous_dim = embed_dim

        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Aggregate layer-wise SSL features into a fixed speaker embedding."""
        # x: [B, D, T, L]
        x = GradMultiply.apply(x, self.feature_grad_mult)

        # Weighted sum across layers
        k = torch.sum(x.mul(F.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(F.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        att_k = self.att_head(k)  # B, T, H

        v = v.unsqueeze(-2)  # B, T, 1, D_c

        pooling_outs = torch.sum(
            v.mul(F.softmax(att_k, dim=1).unsqueeze(-1)), dim=1
        )  # B, H, D_c

        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        outs = self.pooling_fc(pooling_outs)
        return outs


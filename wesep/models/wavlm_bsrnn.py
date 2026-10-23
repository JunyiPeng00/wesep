"""BSRNN extractor blocks with speaker-conditioned fusion.

Used as the extractor of WavLMDSTBSRNN (wesep/models/wavlm_dst_bsrnn.py):
band-split RNN blocks (BSNet) interleaved with speaker fusion layers (FiLM).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from wesep.modules.common.speaker import SpeakerFuseLayer


class ResRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True) -> None:
        super().__init__()
        eps = torch.finfo(torch.float32).eps
        self.norm = nn.GroupNorm(1, input_size, eps)
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * 2 if bidirectional else hidden_size
        self.proj = nn.Linear(out_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # shape: x [B, C, T]
        assert x.ndim == 3, f"Expected [B, C, T], got {tuple(x.shape)}"
        rnn_in = self.norm(x).transpose(1, 2).contiguous()  # shape: [B, T, C]
        if rnn_in.is_cuda:
            # Run the LSTM with the cuDNN/MIOpen fused kernels disabled (kept for
            # parity with the reported runs); the surrounding autocast dtype is kept.
            with torch.backends.cudnn.flags(enabled=False):
                rnn_out, _ = self.rnn(rnn_in)  # shape: [B, T, H]
        else:
            rnn_out, _ = self.rnn(rnn_in)  # shape: [B, T, H]
        proj_out = self.proj(rnn_out).transpose(1, 2).contiguous()  # shape: [B, C, T]
        return x + proj_out


class BSNet(nn.Module):
    def __init__(self, in_channel: int, nband: int, bidirectional: bool = True) -> None:
        super().__init__()
        assert in_channel % nband == 0, "in_channel must be divisible by nband"
        self.nband = nband
        self.feature_dim = in_channel // nband
        self.band_rnn = ResRNN(
            self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional
        )
        self.band_comm = ResRNN(
            self.feature_dim, self.feature_dim * 2, bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor, _dummy: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        # shape: x [B, nband*D, T]
        bsz, channels, t_steps = x.shape
        assert channels == self.nband * self.feature_dim
        band_out = self.band_rnn(x.view(bsz * self.nband, self.feature_dim, -1))
        band_out = band_out.view(bsz, self.nband, self.feature_dim, t_steps)  # [B, nband, D, T]
        comm_in = band_out.permute(0, 3, 2, 1).contiguous().view(
            bsz * t_steps, self.feature_dim, self.nband
        )  # [B*T, D, nband]
        comm_out = self.band_comm(comm_in).view(bsz, t_steps, self.feature_dim, self.nband)
        comm_out = comm_out.permute(0, 3, 2, 1).contiguous()  # [B, nband, D, T]
        return comm_out.view(bsz, channels, t_steps)


class FuseSeparation(nn.Module):
    def __init__(
        self,
        nband: int,
        num_repeat: int,
        feature_dim: int,
        spk_emb_dim: int,
        spk_fuse_type: str,
        multi_fuse: bool,
        use_bidirectional: bool,
    ) -> None:
        super().__init__()
        self.multi_fuse = multi_fuse
        self.nband = nband
        self.feature_dim = feature_dim
        self.separation = nn.ModuleList()
        if self.multi_fuse:
            for _ in range(num_repeat):
                self.separation.append(
                    SpeakerFuseLayer(
                        embed_dim=spk_emb_dim,
                        feat_dim=feature_dim,
                        fuse_type=spk_fuse_type,
                    )
                )
                self.separation.append(
                    BSNet(nband * feature_dim, nband, bidirectional=use_bidirectional)
                )
        else:
            self.separation.append(
                SpeakerFuseLayer(
                    embed_dim=spk_emb_dim,
                    feat_dim=feature_dim,
                    fuse_type=spk_fuse_type,
                )
            )
            for _ in range(num_repeat):
                self.separation.append(
                    BSNet(nband * feature_dim, nband, bidirectional=use_bidirectional)
                )

    def forward(self, x: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        # shape: x [B, nband, D, T], spk_embedding [B, 1, E, 1]
        assert x.ndim == 4, f"Expected [B, nband, D, T], got {tuple(x.shape)}"
        bsz = x.shape[0]
        nch = 1
        if self.multi_fuse:
            for i, module in enumerate(self.separation):
                x = module(x, spk_embedding)
                if i % 2 == 0:
                    x = x.view(bsz * nch, self.nband * self.feature_dim, -1)
                else:
                    x = x.view(bsz * nch, self.nband, self.feature_dim, -1)
        else:
            x = self.separation[0](x, spk_embedding)
            x = x.view(bsz * nch, self.nband * self.feature_dim, -1)
            for idx, module in enumerate(self.separation):
                if idx > 0:
                    x = module(x, spk_embedding)
            x = x.view(bsz * nch, self.nband, self.feature_dim, -1)
        return x



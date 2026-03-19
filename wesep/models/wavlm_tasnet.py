"""Speaker-conditioned separation model using WavLM + TasNet in wesep.

Architecture:
    - Time-domain path: DeepEncoder (10 ms hop) -> mask -> DeepDecoder.
    - Separator path: WavLM (20 ms frame) + learnable layer weighted sum
      + 1× TCNBlock -> upsample to encoder resolution.
    - Speaker path: shared WavLM + MHFA -> speaker embedding; fused with
      separator features and passed through (R * X) TCNBlocks to predict masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.tasnet.convs import Conv1D
from wesep.modules.dpccn.convs import TCNBlock
from wesep.modules.common.speaker import SpeakerTransform, SpeakerFuseLayer
from wesep.modules.wavlm_frontend import (
    HuggingfaceFrontendWavLM,
    WavLMFrontendConfig,
)
from wesep.modules.ssl_backend import SSL_BACKEND_MHFA


@dataclass
class WavLMTasNetConfig:
    """Configuration for WavLMTasNet model.

    Attributes:
        wavlm_name:
            Name of the upstream WavLM variant (e.g. "wavlm_base").
        wavlm_ckpt:
            Path to locally converted WavLM checkpoint with keys
            ``\"config\"`` and ``\"state_dict\"``.
        wavlm_frozen:
            Whether to freeze WavLM parameters.
        encoder_dim:
            Number of channels in TasNet encoder (`N`).
        kernel_size:
            Encoder kernel size in samples.
        stride:
            Encoder stride in samples (hop). For 16 kHz and 10 ms hop,
            use `kernel_size=320`, `stride=160`.
        sep_tcn_channels:
            Channel dimension used in separator and speaker TCN blocks.
        post_fuse_tcn_X:
            Number of TCN blocks in each repeat after speaker fusion
            (aligned with ConvTasNet ``X``).
        post_fuse_tcn_R:
            Number of repeats for post-fusion TCN stack (aligned with
            ConvTasNet ``R``). Dilation is reset in each repeat.
        spk_emb_dim:
            Output dimensionality of MHFA speaker embedding.
    """

    wavlm_name: str = "wavlm_base"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True
    encoder_dim: int = 512
    kernel_size: int = 320
    stride: int = 160
    sep_tcn_channels: int = 256
    post_fuse_tcn_X: int = 3
    post_fuse_tcn_R: int = 1
    spk_emb_dim: int = 256


class LayerWeightedSum(nn.Module):
    """Learnable layer-wise weighted sum over WavLM hidden states.

    Input:
        x: Tensor of shape [B, C, T, L].
    Output:
        Tensor of shape [B, C, T].
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.alpha = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 4 or x.shape[-1] != self.num_layers:
            raise ValueError(
                f"Expected input of shape [B, C, T, L={self.num_layers}], got {tuple(x.shape)}"
            )
        weights = F.softmax(self.alpha, dim=-1)
        # [B, C, T, L] * [L] -> [B, C, T]
        return torch.sum(x * weights.view(1, 1, 1, -1), dim=-1)


class WavLMTasNet(nn.Module):
    """Speaker-conditioned separation model combining WavLM and TasNet."""

    def __init__(self, cfg: Optional[WavLMTasNetConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            for key in ("joint_training", "multi_task"):
                kwargs.pop(key, None)
            cfg = WavLMTasNetConfig(**kwargs)
        self.cfg = cfg
        if cfg.kernel_size <= 0 or cfg.stride <= 0:
            raise ValueError(
                f"kernel_size and stride must be positive, got kernel_size={cfg.kernel_size}, stride={cfg.stride}"
            )
        if cfg.kernel_size < cfg.stride:
            raise ValueError(
                f"kernel_size should be >= stride for stable overlap-add style reconstruction, "
                f"got kernel_size={cfg.kernel_size}, stride={cfg.stride}"
            )

        # WavLM frontend (shared for mixture and reference)
        wavlm_cfg = WavLMFrontendConfig(
            name=cfg.wavlm_name,
            path_or_url=cfg.wavlm_ckpt,
            frozen=cfg.wavlm_frozen,
        )
        self.ssl_frontend = HuggingfaceFrontendWavLM(wavlm_cfg)
        feat_dim = self.ssl_frontend.output_size()

        # MHFA speaker backend
        # Derive layer count directly from the loaded model structure.
        nb_layer = len(self.ssl_frontend.upstream.encoder.transformer.layers) + 1
        self.mhfa_backend = SSL_BACKEND_MHFA(
            head_nb=8,
            feat_dim=feat_dim,
            compression_dim=128,
            embed_dim=cfg.spk_emb_dim,
            nb_layer=nb_layer,
            feature_grad_mult=1.0,
        )

        # Layer-weighted sum over mixture SSL features
        self.layer_weighted_sum = LayerWeightedSum(num_layers=nb_layer)

        # Separator-side TCN stack on 20 ms SSL features.
        self.sep_tcn_20ms = nn.Sequential(
            TCNBlock(feat_dim, feat_dim, kernel_size=3, dilation=1, causal=False),
            TCNBlock(feat_dim, feat_dim, kernel_size=3, dilation=2, causal=False),
            TCNBlock(feat_dim, feat_dim, kernel_size=3, dilation=4, causal=False),
            TCNBlock(feat_dim, feat_dim, kernel_size=3, dilation=8, causal=False),
        )

        # Project SSL features to separator TCN channels for fusion
        self.ssl_proj = Conv1D(feat_dim, cfg.sep_tcn_channels, kernel_size=1)

        # Learnable 2× upsampling: 20 ms → 10 ms resolution.
        # Initialized to bilinear interpolation so the layer is identity-like
        # at the start of training, then learns to preserve transient details.
        self.upsample = nn.ConvTranspose1d(
            cfg.sep_tcn_channels, cfg.sep_tcn_channels,
            kernel_size=4, stride=2, padding=1, bias=False,
        )
        self._init_upsample_bilinear()

        # Speaker transform and FiLM fusion
        self.spk_transform = SpeakerTransform(embed_dim=cfg.spk_emb_dim)
        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=cfg.spk_emb_dim,
            feat_dim=cfg.sep_tcn_channels,
            fuse_type="FiLM",
        )

        post_fuse_x = cfg.post_fuse_tcn_X
        post_fuse_r = cfg.post_fuse_tcn_R

        if post_fuse_x <= 0:
            raise ValueError(
                f"post_fuse_tcn_X must be > 0, got {post_fuse_x}"
            )
        if post_fuse_r <= 0:
            raise ValueError(f"post_fuse_tcn_R must be > 0, got {post_fuse_r}")

        # Configurable TCN stack after speaker fusion at 10 ms resolution.
        # Aligned with ConvTasNet X/R design:
        # each repeat uses dilations 1, 2, 4, ... and resets at next repeat.
        post_fuse_blocks = []
        for _ in range(post_fuse_r):
            for block_idx in range(post_fuse_x):
                post_fuse_blocks.append(
                    TCNBlock(
                        in_dims=cfg.sep_tcn_channels,
                        out_dims=cfg.sep_tcn_channels,
                        kernel_size=3,
                        dilation=2**block_idx,
                        causal=False,
                    )
                )
        self.post_fuse_tcn = nn.Sequential(*post_fuse_blocks)

        # Mask generator: separator channels -> encoder_dim
        self.mask_gen = Conv1D(cfg.sep_tcn_channels, cfg.encoder_dim, kernel_size=1)

        # Single-layer time-domain analysis/synthesis CNN pair.
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.encoder_dim,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            bias=False,
        )
        self.decoder = nn.ConvTranspose1d(
            in_channels=cfg.encoder_dim,
            out_channels=1,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            bias=False,
        )

    @torch.no_grad()
    def _init_upsample_bilinear(self) -> None:
        """Initialize ConvTranspose1d to bilinear interpolation so training
        starts from the same point as F.interpolate(mode='linear')."""
        w = self.upsample.weight  # [in_C, out_C, kernel_size]
        w.zero_()
        k = w.shape[-1]
        factor = (k + 1) // 2
        center = factor - 0.5 if k % 2 == 0 else factor - 1.0
        filt = 1.0 - torch.abs(torch.arange(k, dtype=w.dtype) - center) / factor
        for i in range(w.shape[0]):
            w[i, i, :] = filt

    def _encode_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform with single-layer Conv1d, ensuring channel dimension."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.ndim == 3, f"Expected encoder input [B, 1, T], got {tuple(x.shape)}"
        return self.encoder(x)

    def _decode_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """Decode encoder-domain representation with single-layer ConvTranspose1d."""
        assert x.ndim == 3, f"Expected decoder input [B, N, T_enc], got {tuple(x.shape)}"
        y = self.decoder(x)  # shape: [B, 1, T_out]
        return y.squeeze(1)  # shape: [B, T_out]

    def forward(
        self,
        mix_wav: torch.Tensor,
        ref_wav: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            mix_wav:
                Mixture waveform tensor of shape [B, T].
            ref_wav:
                Reference speaker waveform tensor of shape [B, T_ref].

        Returns:
            Estimated target waveform of shape [B, T_out], where T_out is
            aligned to the mixture length.
        """
        if mix_wav.dim() != 2 or ref_wav.dim() != 2:
            raise ValueError("mix_wav and ref_wav must be 2-D tensors [B, T].")

        batch_size, num_samples = mix_wav.shape

        # 1) Time-domain encoder (10 ms stride)
        enc_mix = self._encode_waveform(mix_wav)  # [B, N, T_10]

        # 2) SSL features for mixture and reference
        ssl_mix, _ = self.ssl_frontend(mix_wav, None)  # [B, C, T_20, L]
        ssl_ref, _ = self.ssl_frontend(ref_wav, None)  # [B, C, T_20_ref, L]

        # Ensure same number of layers for weighted sum / MHFA
        # ssl_frontend returns layers including pre-transformer; MHFA expects L=nb_layer.
        # We simply pass the full stack.

        # 3) Mixture separator path: layer-wise sum + TCN + upsample to 10 ms
        feat_mix_20ms = self.layer_weighted_sum(ssl_mix)  # [B, C, T_20]
        feat_mix_20ms = self.sep_tcn_20ms(feat_mix_20ms)  # [B, C, T_20]

        # Project to separator channels
        sep_feat_20ms = self.ssl_proj(feat_mix_20ms)  # [B, sep_C, T_20]

        # Learnable 2x upsample to encoder time resolution
        t_enc = enc_mix.size(-1)
        sep_feat_10ms = self.upsample(sep_feat_20ms)  # [B, sep_C, ~2*T_20]
        if sep_feat_10ms.size(-1) > t_enc:
            sep_feat_10ms = sep_feat_10ms[..., :t_enc]
        elif sep_feat_10ms.size(-1) < t_enc:
            sep_feat_10ms = F.pad(
                sep_feat_10ms, (0, t_enc - sep_feat_10ms.size(-1))
            )

        # 4) Speaker path: MHFA on reference SSL features
        spk_emb = self.mhfa_backend(ssl_ref)  # [B, spk_emb_dim]
        spk_emb = self.spk_transform(spk_emb)  # [B, spk_emb_dim]

        # 5) Speaker fusion at 10 ms resolution
        spk_emb_unsq = spk_emb.unsqueeze(-1)  # [B, spk_emb_dim, 1]
        fused_feat = self.spk_fuse(sep_feat_10ms, spk_emb_unsq)  # [B, sep_C, T_10]

        # 6) Post-fusion TCN stack
        h = self.post_fuse_tcn(fused_feat)  # [B, sep_C, T_10]

        # 7) Mask generation and application in encoder domain.
        m = F.relu(self.mask_gen(h))  # [B, N, T_10]
        # Align encoder temporal dimension with mask via interpolation if needed
        if m.size(-1) != enc_mix.size(-1):
            m = F.interpolate(m, size=enc_mix.size(-1), mode="linear", align_corners=False)

        masked_enc = enc_mix * m

        # 8) Decode back to waveform
        est = self._decode_waveform(masked_enc)  # shape: [B, T_out]

        # Crop/pad to match mixture length
        if est.size(-1) > num_samples:
            est = est[..., :num_samples]
        elif est.size(-1) < num_samples:
            pad_len = num_samples - est.size(-1)
            est = F.pad(est, (0, pad_len))

        return est


def build_wavlm_tasnet(config: Optional[WavLMTasNetConfig] = None) -> WavLMTasNet:
    """Factory helper to build WavLMTasNet from a config."""
    if config is None:
        config = WavLMTasNetConfig()
    return WavLMTasNet(config)


if __name__ == "__main__":
    # Minimal sanity check with forward + backward.
    model = WavLMTasNet(
        wavlm_name="wavlm_base",
        wavlm_ckpt="/path/to/convert/wavlm_base_plus.hf.pth",
        wavlm_frozen=True,
    )
    x = torch.randn(2, 32000)
    ref = torch.randn(2, 32000)
    target = torch.randn(2, 32000)
    y = model(x, ref)
    loss = F.l1_loss(y, target)
    loss.backward()
    encoder_sparsity = float((model.encoder.weight == 0).sum().item()) / float(
        model.encoder.weight.numel()
    )
    print(y.shape)
    print(f"loss={loss.item():.6f}")
    print(f"encoder_weight_sparsity={encoder_sparsity:.6f}")


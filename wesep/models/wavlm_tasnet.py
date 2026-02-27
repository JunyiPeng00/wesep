"""Speaker-conditioned separation model using WavLM + TasNet in wesep.

Architecture:
    - Time-domain path: DeepEncoder (10 ms hop) -> mask -> DeepDecoder.
    - Separator path: WavLM (20 ms frame) + learnable layer weighted sum
      + 1× TCNBlock -> upsample to encoder resolution.
    - Speaker path: shared WavLM + MHFA -> speaker embedding; fused with
      separator features and passed through 3× TCNBlock to predict masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.tasnet import DeepEncoder, DeepDecoder
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
        bottleneck_dim:
            Bottleneck dimension after encoder (`B`).
        kernel_size:
            Encoder kernel size in samples.
        stride:
            Encoder stride in samples (hop). For 16 kHz and 10 ms hop,
            use `kernel_size=320`, `stride=160`.
        sep_tcn_channels:
            Channel dimension used in separator and speaker TCN blocks.
        spk_emb_dim:
            Output dimensionality of MHFA speaker embedding.
    """

    wavlm_name: str = "wavlm_base"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True
    encoder_dim: int = 512
    bottleneck_dim: int = 256
    kernel_size: int = 320
    stride: int = 160
    sep_tcn_channels: int = 256
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

    def __init__(self, cfg: WavLMTasNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # WavLM frontend (shared for mixture and reference)
        wavlm_cfg = WavLMFrontendConfig(
            name=cfg.wavlm_name,
            path_or_url=cfg.wavlm_ckpt,
            frozen=cfg.wavlm_frozen,
        )
        self.ssl_frontend = HuggingfaceFrontendWavLM(wavlm_cfg)
        feat_dim = self.ssl_frontend.output_size()

        # MHFA speaker backend
        # Number of layers is taken from saved config; default 13 if absent.
        nb_layer = len(self.ssl_frontend.upstream_config.get("encoder_use_attention", [])) + 1
        if nb_layer <= 1:
            nb_layer = 13
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

        # Separator-side TCN on 20 ms SSL features
        self.sep_tcn_20ms = TCNBlock(
            in_dims=feat_dim,
            out_dims=feat_dim,
            kernel_size=3,
            dilation=1,
            causal=False,
        )

        # Project SSL features to separator TCN channels for fusion
        self.ssl_proj = Conv1D(feat_dim, cfg.sep_tcn_channels, kernel_size=1)

        # Speaker transform and fusion
        self.spk_transform = SpeakerTransform(embed_dim=cfg.spk_emb_dim)
        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=cfg.spk_emb_dim,
            feat_dim=cfg.sep_tcn_channels,
            fuse_type="multiply",
        )

        # 3× TCN blocks after speaker fusion at 10 ms resolution
        self.post_fuse_tcn = nn.Sequential(
            TCNBlock(
                in_dims=cfg.sep_tcn_channels,
                out_dims=cfg.sep_tcn_channels,
                kernel_size=3,
                dilation=1,
                causal=False,
            ),
            TCNBlock(
                in_dims=cfg.sep_tcn_channels,
                out_dims=cfg.sep_tcn_channels,
                kernel_size=3,
                dilation=2,
                causal=False,
            ),
            TCNBlock(
                in_dims=cfg.sep_tcn_channels,
                out_dims=cfg.sep_tcn_channels,
                kernel_size=3,
                dilation=4,
                causal=False,
            ),
        )

        # Mask generator: separator channels -> encoder_dim
        self.mask_gen = Conv1D(cfg.sep_tcn_channels, cfg.encoder_dim, kernel_size=1)

        # TasNet encoder / decoder
        self.encoder = DeepEncoder(
            in_channels=1,
            out_channels=cfg.encoder_dim,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
        )
        self.decoder = DeepDecoder(
            N=cfg.encoder_dim,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
        )

    def _encode_waveform(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform with DeepEncoder, ensuring channel dimension."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

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

        # Upsample to encoder time resolution
        t_enc = enc_mix.size(-1)
        sep_feat_10ms = F.interpolate(
            sep_feat_20ms,
            size=t_enc,
            mode="linear",
            align_corners=False,
        )  # [B, sep_C, T_10]

        # 4) Speaker path: MHFA on reference SSL features
        spk_emb = self.mhfa_backend(ssl_ref)  # [B, spk_emb_dim]
        spk_emb = self.spk_transform(spk_emb)  # [B, spk_emb_dim]

        # 5) Speaker fusion at 10 ms resolution
        spk_emb_unsq = spk_emb.unsqueeze(-1)  # [B, spk_emb_dim, 1]
        fused_feat = self.spk_fuse(sep_feat_10ms, spk_emb_unsq)  # [B, sep_C, T_10]

        # 6) Post-fusion TCN stack
        h = self.post_fuse_tcn(fused_feat)  # [B, sep_C, T_10]

        # 7) Mask generation and application in encoder domain
        m = torch.sigmoid(self.mask_gen(h))  # [B, N, T_10]
        # Align encoder temporal dimension with mask via interpolation if needed
        if m.size(-1) != enc_mix.size(-1):
            m = F.interpolate(m, size=enc_mix.size(-1), mode="linear", align_corners=False)

        masked_enc = enc_mix * m

        # 8) Decode back to waveform
        est = self.decoder(masked_enc)  # [B, T_out]

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
    # Minimal sanity check
    cfg = WavLMTasNetConfig(
        wavlm_name="wavlm_base",
        wavlm_ckpt="/Users/pengjy/Interspeech2026/DynamicPruning/wespeaker_hubert/examples/voxceleb/v4_pruning/convert/wavlm_base.hf.pth",
        wavlm_frozen=True,
    )
    model = WavLMTasNet(cfg)
    x = torch.randn(2, 32000)
    ref = torch.randn(2, 32000)
    with torch.no_grad():
        y = model(x, ref)
    print(y.shape)


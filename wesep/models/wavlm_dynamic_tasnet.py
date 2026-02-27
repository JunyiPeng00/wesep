"""WavLM-based dynamic TSE: single WavLM, input enroll | 1s silence | mix."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.tasnet import DeepEncoder, DeepDecoder
from wesep.modules.tasnet.convs import Conv1D
from wesep.modules.dpccn.convs import TCNBlock
from wesep.modules.wavlm_frontend import WavLMFrontendConfig
from wesep.modules.wavlm_frontend.model import wav2vec2_model

logger = logging.getLogger(__name__)

WAVLM_FRAME_STRIDE = 320
ATTN_MASK_NEG_INF = -1e9


@dataclass
class WavLMDynamicTasNetConfig:
    wavlm_name: str = "wavlm_base"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True

    encoder_dim: int = 512
    bottleneck_dim: int = 256
    kernel_size: int = 320
    stride: int = 160

    sep_tcn_channels: int = 256
    num_tcn_layers: int = 3
    num_hybrid_blocks: int = 5
    sample_rate: int = 16000
    silence_seconds: float = 1.0
    frame_stride: int = WAVLM_FRAME_STRIDE
    qkb_threshold: float = 0.5
    hybrid_mask_asymmetric: bool = False


class LayerWeightedSum(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_list: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=-1)
        out = sum(w * h for w, h in zip(weights, layer_list))
        return out


def _compute_qkb_bias(
    x_mix: torch.Tensor,
    spk_emb: torch.Tensor,
    threshold: Union[float, torch.Tensor],
) -> torch.Tensor:
    B, T_mix, C = x_mix.shape
    x_n = F.normalize(x_mix, dim=-1)
    spk_n = F.normalize(spk_emb, dim=-1)
    cos_sim = torch.einsum("btc,bc->bt", x_n, spk_n)
    s_t = torch.sigmoid(cos_sim)
    M = s_t < threshold
    bias = torch.zeros(B, 1, T_mix, T_mix, device=x_mix.device, dtype=x_mix.dtype)
    bias = bias.masked_fill(M.unsqueeze(1).unsqueeze(2).expand(B, 1, T_mix, T_mix), ATTN_MASK_NEG_INF)
    return bias


def _build_hybrid_attention_mask(
    T: int,
    enroll_end_frame: int,
    mix_start_frame: int,
    device: torch.device,
    dtype: torch.dtype,
    asymmetric: bool = False,
) -> torch.Tensor:
    i_idx = torch.arange(T, device=device, dtype=torch.long)
    j_idx = torch.arange(T, device=device, dtype=torch.long)
    if asymmetric:
        allow = ((i_idx.unsqueeze(1) < enroll_end_frame) & (j_idx.unsqueeze(0) < enroll_end_frame)) | (
            i_idx.unsqueeze(1) >= enroll_end_frame
        )
    else:
        in_enroll = (i_idx.unsqueeze(1) < enroll_end_frame) & (j_idx.unsqueeze(0) < enroll_end_frame)
        in_silence = (
            (i_idx.unsqueeze(1) >= enroll_end_frame) & (i_idx.unsqueeze(1) < mix_start_frame)
            & (j_idx.unsqueeze(0) >= enroll_end_frame) & (j_idx.unsqueeze(0) < mix_start_frame)
        )
        in_mix = (i_idx.unsqueeze(1) >= mix_start_frame) & (j_idx.unsqueeze(0) >= mix_start_frame)
        allow = in_enroll | in_silence | in_mix
    mask = torch.where(allow.unsqueeze(0).unsqueeze(0), torch.zeros(1, 1, T, T, device=device, dtype=dtype), torch.full((1, 1, T, T), ATTN_MASK_NEG_INF, device=device, dtype=dtype))
    return mask


class WavLMDynamicSeparator(nn.Module):
    def __init__(self, cfg: WavLMDynamicTasNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if not cfg.wavlm_ckpt:
            raise ValueError("WavLMDynamicSeparator requires wavlm_ckpt pointing to a converted WavLM .pth file.")
        state = torch.load(cfg.wavlm_ckpt, map_location="cpu")
        upstream_config = state["config"]
        upstream_config.update(
            dict(
                extractor_prune_conv_channels=False,
                encoder_prune_attention_heads=False,
                encoder_prune_attention_layer=False,
                encoder_prune_feed_forward_intermediate=False,
                encoder_prune_feed_forward_layer=False,
            )
        )
        self.upstream = wav2vec2_model(**upstream_config)
        self.upstream.load_state_dict(state["state_dict"], strict=False)
        if cfg.wavlm_frozen:
            for p in self.upstream.parameters():
                p.requires_grad_(False)

        self.feat_dim = self._output_size()
        self.num_layers = len(self.upstream.encoder.transformer.layers)
        self.num_hybrid = min(cfg.num_hybrid_blocks, self.num_layers)
        self.layer_weight = LayerWeightedSum(self.num_layers + 1)
        self.qkb_threshold = float(cfg.qkb_threshold)
        self.hybrid_mask_asymmetric = getattr(cfg, "hybrid_mask_asymmetric", False)
        self.mix_proj = Conv1D(self.feat_dim, cfg.sep_tcn_channels, kernel_size=1)
        self.tcn = nn.Sequential(*[
            TCNBlock(
                in_dims=cfg.sep_tcn_channels,
                out_dims=cfg.sep_tcn_channels,
                kernel_size=3,
                dilation=2**i,
                causal=False,
            )
            for i in range(cfg.num_tcn_layers)
        ])
        self.mask_head = Conv1D(cfg.sep_tcn_channels, cfg.encoder_dim, kernel_size=1)

    def _output_size(self) -> int:
        return 1024 if "large" in self.cfg.wavlm_name.lower() else 768

    def _get_frame_indices(
        self,
        T_frames: int,
        enroll_len: int,
        mix_start: int,
    ) -> Tuple[int, int]:
        stride = self.cfg.frame_stride
        enroll_end_frame = min(enroll_len // stride, T_frames)
        mix_start_frame = min(mix_start // stride, T_frames)
        return enroll_end_frame, mix_start_frame

    def forward(
        self,
        wav: torch.Tensor,
        enroll_len: int,
        mix_start: int,
        enc_mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, T_enc = enc_mix.shape

        with torch.no_grad() if self.cfg.wavlm_frozen else contextlib.nullcontext():
            feat, _ = self.upstream.feature_extractor(wav, None)
        T_frames = feat.size(1)
        enroll_end_frame, mix_start_frame = self._get_frame_indices(T_frames, enroll_len, mix_start)

        x, pad_mask = self.upstream.encoder._preprocess(feat, None)
        hybrid_mask = _build_hybrid_attention_mask(
            T_frames, enroll_end_frame, mix_start_frame,
            x.device, x.dtype,
            asymmetric=self.hybrid_mask_asymmetric,
        )
        if pad_mask is not None:
            hybrid_mask = hybrid_mask + pad_mask

        x = self.upstream.encoder.transformer._preprocess(x)
        position_bias = None
        mix_outputs: List[torch.Tensor] = [x[:, mix_start_frame:, :]]

        for i, layer in enumerate(self.upstream.encoder.transformer.layers):
            if i < self.num_hybrid:
                attn_mask = hybrid_mask
                x, position_bias = layer(x, attn_mask, position_bias=position_bias)
                if i == self.num_hybrid - 1:
                    spk_emb = x[:, :enroll_end_frame, :].mean(dim=1)
                    x_mix = x[:, mix_start_frame:, :].contiguous()
                    position_bias = None
                mix_outputs.append(x[:, mix_start_frame:, :])
            else:
                b_qkb = _compute_qkb_bias(x_mix, spk_emb, self.qkb_threshold)
                x_mix, position_bias = layer(x_mix, b_qkb, position_bias=position_bias)
                mix_outputs.append(x_mix)

        mix_feat = self.layer_weight(mix_outputs)
        mix_feat = mix_feat.transpose(1, 2)
        mix_feat = self.mix_proj(mix_feat)
        mix_feat_10 = F.interpolate(mix_feat, size=T_enc, mode="linear", align_corners=False)
        for blk in self.tcn:
            mix_feat_10 = blk(mix_feat_10)
        m = torch.sigmoid(self.mask_head(mix_feat_10))
        if m.size(-1) != T_enc:
            m = F.interpolate(m, size=T_enc, mode="linear", align_corners=False)
        return m, spk_emb

    def get_speaker_embedding_from_wav(
        self,
        wav: torch.Tensor,
        enroll_len: int,
        mix_start: int,
    ) -> torch.Tensor:
        with torch.no_grad() if self.cfg.wavlm_frozen else contextlib.nullcontext():
            feat, _ = self.upstream.feature_extractor(wav, None)
        T_frames = feat.size(1)
        enroll_end_frame, mix_start_frame = self._get_frame_indices(T_frames, enroll_len, mix_start)

        x, pad_mask = self.upstream.encoder._preprocess(feat, None)
        hybrid_mask = _build_hybrid_attention_mask(
            T_frames, enroll_end_frame, mix_start_frame,
            x.device, x.dtype,
            asymmetric=self.hybrid_mask_asymmetric,
        )
        if pad_mask is not None:
            hybrid_mask = hybrid_mask + pad_mask

        x = self.upstream.encoder.transformer._preprocess(x)
        position_bias = None
        for i, layer in enumerate(self.upstream.encoder.transformer.layers):
            if i >= self.num_hybrid:
                break
            attn_mask = hybrid_mask
            x, position_bias = layer(x, attn_mask, position_bias=position_bias)
        spk_emb = x[:, :enroll_end_frame, :].mean(dim=1)
        return spk_emb


class WavLMDynamicTasNet(nn.Module):
    def __init__(self, cfg: WavLMDynamicTasNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

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
        self.separator = WavLMDynamicSeparator(cfg)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

    def _default_mix_start(self, enroll_len: int) -> int:
        return enroll_len + int(self.cfg.silence_seconds * self.cfg.sample_rate)

    def forward_concat(
        self,
        wav: torch.Tensor,
        enroll_len: int,
        mix_start: Optional[int] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if wav.dim() != 2:
            raise ValueError("wav must be a 2-D tensor [B, T].")
        if mix_start is None:
            mix_start = self._default_mix_start(enroll_len)

        mix_wav = wav[:, mix_start:]
        T_m = mix_wav.size(-1)

        enc_mix = self._encode(mix_wav)
        m, spk_emb = self.separator(wav, enroll_len, mix_start, enc_mix)
        masked_enc = enc_mix * m
        est = self.decoder(masked_enc)

        if est.size(-1) > T_m:
            est = est[..., :T_m]
        elif est.size(-1) < T_m:
            est = F.pad(est, (0, T_m - est.size(-1)))

        if return_embedding:
            return est, spk_emb
        return est

    def extract_speaker_embedding(
        self,
        wav: torch.Tensor,
        enroll_len: int,
        mix_start: Optional[int] = None,
    ) -> torch.Tensor:
        if mix_start is None:
            mix_start = self._default_mix_start(enroll_len)
        return self.separator.get_speaker_embedding_from_wav(wav, enroll_len, mix_start)

    def load_pretrained_wavlm(self, ckpt_path: str, strict: bool = False) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in state:
            raise ValueError(f"WavLM ckpt must contain 'state_dict'; got keys: {list(state.keys())}")
        self.separator.upstream.load_state_dict(state["state_dict"], strict=strict)

    def forward(
        self,
        mix_wav: torch.Tensor,
        enroll_wav: torch.Tensor,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T_enroll = enroll_wav.shape
        silence_samples = int(self.cfg.silence_seconds * self.cfg.sample_rate)
        silence = torch.zeros(B, silence_samples, device=mix_wav.device, dtype=mix_wav.dtype)
        wav = torch.cat([enroll_wav, silence, mix_wav], dim=-1)
        return self.forward_concat(wav, enroll_len=T_enroll, mix_start=self._default_mix_start(T_enroll), return_embedding=return_embedding)


def build_wavlm_dynamic_tasnet(
    config: Optional[WavLMDynamicTasNetConfig] = None,
) -> WavLMDynamicTasNet:
    if config is None:
        config = WavLMDynamicTasNetConfig()
    return WavLMDynamicTasNet(config)


def from_pretrained_wavlm(
    wavlm_ckpt_path: str,
    **config_overrides,
) -> WavLMDynamicTasNet:
    cfg = WavLMDynamicTasNetConfig(wavlm_ckpt=wavlm_ckpt_path, **config_overrides)
    return WavLMDynamicTasNet(cfg)

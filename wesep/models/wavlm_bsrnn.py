"""WavLM-conditioned BSRNN for target speaker extraction.

Architecture overview:
    1) Frequency-domain BSRNN branch:
       STFT -> subband split -> per-band bottleneck -> BSNet separator.
    2) WavLM branch:
       - Mixture WavLM hidden states are layer-weighted and projected as
         conditioning features for subband representations.
       - Enrollment WavLM hidden states are aggregated by MHFA to produce
         speaker embedding for speaker-aware fusion.
    3) Complex mask estimation + iSTFT reconstruction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.modules.common.speaker import SpeakerFuseLayer, SpeakerTransform
# from wesep.modules.ssl import AIEConfig, AdaptiveInputEnhancer
from wesep.modules.ssl_backend import SSL_BACKEND_MHFA
from wesep.modules.tasnet.convs import Conv1D
from wesep.modules.wavlm_frontend import (
    HuggingfaceFrontendWavLM,
    WavLMFrontendConfig,
)


@dataclass
class WavLMBSRNNConfig:
    wavlm_name: str = "wavlm_base"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True

    sr: int = 16000
    win: int = 512
    stride: int = 128
    feature_dim: int = 128
    num_repeat: int = 6
    use_bidirectional: bool = True

    spk_emb_dim: int = 256
    spk_fuse_type: str = "multiply"
    multi_fuse: bool = False
    use_spk_transform: bool = True


@dataclass
class WavLMAIEBSRNNConfig:
    wavlm_name: str = "wavlm_base"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True

    sr: int = 16000
    win: int = 512
    stride: int = 128
    feature_dim: int = 128
    num_repeat: int = 6
    use_bidirectional: bool = True

    spk_emb_dim: int = 256
    spk_fuse_type: str = "multiply"
    multi_fuse: bool = False
    use_spk_transform: bool = True

    aie_enable: bool = True
    aie_mode: str = "fpm"
    aie_hidden_dim: int = 256
    aie_use_transformer: bool = True
    aie_use_cnn: bool = True
    aie_selected_cnn_levels: Union[str, Sequence[int]] = "auto"
    mix_fuse_type: str = "add"
    use_shared_ssl_for_mix_and_ref: bool = True


class LayerWeightedSum(nn.Module):
    """Learnable layer-wise weighted sum.

    Input shape: [B, C, T, L]
    Output shape: [B, C, T]
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        assert num_layers > 0, f"num_layers must be > 0, got {num_layers}"
        self.num_layers = num_layers
        self.alpha = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        assert x.ndim == 4, f"Expected 4D tensor [B, C, T, L], got {x.ndim}D"
        assert x.shape[-1] == self.num_layers, (
            f"Layer mismatch: input L={x.shape[-1]}, expected {self.num_layers}"
        )
        # shape: [L]
        weights = F.softmax(self.alpha, dim=-1)
        # shape: [B, C, T]
        return torch.sum(x * weights.view(1, 1, 1, -1), dim=-1)


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


class WavLMBSRNN(nn.Module):
    def __init__(self, cfg: Optional[WavLMBSRNNConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            for key in ("joint_training", "multi_task", "speaker_feat"):
                kwargs.pop(key, None)
            cfg = WavLMBSRNNConfig(**kwargs)
        self.cfg = cfg

        assert cfg.win > 0 and cfg.stride > 0, "win and stride must be positive"
        assert cfg.win >= cfg.stride, "win should be >= stride"

        self.enc_dim = cfg.win // 2 + 1
        self.band_width = self._build_bandwidth(cfg.sr, self.enc_dim)
        self.nband = len(self.band_width)

        if not cfg.wavlm_ckpt:
            raise ValueError("wavlm_ckpt must be provided for WavLMBSRNN")
        wavlm_cfg = WavLMFrontendConfig(
            name=cfg.wavlm_name, path_or_url=cfg.wavlm_ckpt, frozen=cfg.wavlm_frozen
        )
        self.ssl_frontend = HuggingfaceFrontendWavLM(wavlm_cfg)
        feat_dim = self.ssl_frontend.output_size()
        nb_layer = len(self.ssl_frontend.upstream.encoder.transformer.layers) + 1

        self.layer_weight = LayerWeightedSum(nb_layer)
        self.mhfa_backend = SSL_BACKEND_MHFA(
            head_nb=8,
            feat_dim=feat_dim,
            compression_dim=128,
            embed_dim=cfg.spk_emb_dim,
            nb_layer=nb_layer,
            feature_grad_mult=1.0,
        )
        self.spk_transform = SpeakerTransform(embed_dim=cfg.spk_emb_dim)
        self.mix_proj = Conv1D(feat_dim, cfg.feature_dim, kernel_size=1)

        eps = torch.finfo(torch.float32).eps
        self.band_bottlenecks = nn.ModuleList()
        for bw in self.band_width:
            self.band_bottlenecks.append(
                nn.Sequential(
                    nn.GroupNorm(1, bw * 2, eps),
                    nn.Conv1d(bw * 2, cfg.feature_dim, kernel_size=1),
                )
            )

        self.separator = FuseSeparation(
            nband=self.nband,
            num_repeat=cfg.num_repeat,
            feature_dim=cfg.feature_dim,
            spk_emb_dim=cfg.spk_emb_dim,
            spk_fuse_type=cfg.spk_fuse_type,
            multi_fuse=cfg.multi_fuse,
            use_bidirectional=cfg.use_bidirectional,
        )

        self.mask_heads = nn.ModuleList()
        for bw in self.band_width:
            self.mask_heads.append(
                nn.Sequential(
                    nn.GroupNorm(1, cfg.feature_dim, eps),
                    nn.Conv1d(cfg.feature_dim, cfg.feature_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(cfg.feature_dim * 4, cfg.feature_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(cfg.feature_dim * 4, bw * 4, kernel_size=1),
                )
            )

    @staticmethod
    def _build_bandwidth(sr: int, enc_dim: int) -> List[int]:
        b100 = int(np.floor(100 / (sr / 2.0) * enc_dim))
        b200 = int(np.floor(200 / (sr / 2.0) * enc_dim))
        b500 = int(np.floor(500 / (sr / 2.0) * enc_dim))
        b2k = int(np.floor(2000 / (sr / 2.0) * enc_dim))
        widths = [b100] * 15 + [b200] * 10 + [b500] * 5 + [b2k] * 1
        widths.append(enc_dim - int(np.sum(widths)))
        assert sum(widths) == enc_dim, "Subband widths must sum to enc_dim"
        return widths

    def _extract_stft_subbands(
        self,
        mix_wav: torch.Tensor,
    ) -> tuple[int, List[torch.Tensor], List[torch.Tensor]]:
        """Compute STFT and split it into subbands."""
        bsz, _ = mix_wav.shape
        # shape: [B, F, T_spec]
        spec = torch.stft(
            mix_wav,
            n_fft=self.cfg.win,
            hop_length=self.cfg.stride,
            window=torch.hann_window(
                self.cfg.win, dtype=mix_wav.dtype, device=mix_wav.device
            ),
            return_complex=True,
        )
        t_spec = spec.shape[-1]
        # shape: [B, 2, F, T_spec]
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)

        subband_spec_ri: List[torch.Tensor] = []
        subband_mix_complex: List[torch.Tensor] = []
        band_idx = 0
        for bw in self.band_width:
            subband_spec_ri.append(spec_ri[:, :, band_idx:band_idx + bw].contiguous())
            subband_mix_complex.append(spec[:, band_idx:band_idx + bw])
            band_idx += bw
        assert band_idx == self.enc_dim, (
            f"Subband split ended at {band_idx}, expected enc_dim={self.enc_dim}"
        )
        assert len(subband_spec_ri) == self.nband == len(subband_mix_complex)
        return t_spec, subband_spec_ri, subband_mix_complex

    def _encode_reference_hidden_states(self, ref_wav: torch.Tensor) -> torch.Tensor:
        """Encode reference waveform with the speaker SSL frontend."""
        ssl_frontend = getattr(self, "ref_ssl_frontend", None)
        if ssl_frontend is None:
            ssl_frontend = self.ssl_frontend
        ssl_ref, _ = ssl_frontend(ref_wav, None)  # shape: [B, C_ssl, T_ssl, L]
        assert ssl_ref.ndim == 4, f"Expected reference SSL stack [B, C, T, L], got {tuple(ssl_ref.shape)}"
        return ssl_ref

    def _compute_speaker_embedding(self, ref_wav: torch.Tensor) -> torch.Tensor:
        """Compute speaker embedding and reshape for separator fusion."""
        ssl_ref = self._encode_reference_hidden_states(ref_wav)
        spk_emb = self.mhfa_backend(ssl_ref)  # shape: [B, E]
        if getattr(self.cfg, "use_spk_transform", True):
            spk_emb = self.spk_transform(spk_emb)  # shape: [B, E]
        return spk_emb.unsqueeze(1).unsqueeze(3)  # shape: [B, 1, E, 1]

    def _build_subband_feature_tensor(
        self,
        subband_spec_ri: List[torch.Tensor],
        mix_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse subband bottlenecks with the mixture conditioning feature."""
        assert mix_cond.ndim == 3, f"Expected mix_cond [B, D, T_spec], got {tuple(mix_cond.shape)}"
        bsz = mix_cond.shape[0]
        subband_features: List[torch.Tensor] = []
        for i, bn in enumerate(self.band_bottlenecks):
            bw = self.band_width[i]
            # shape: [B, 2 * bw, T_spec]
            this_feat = subband_spec_ri[i].view(bsz, bw * 2, -1)
            # shape: [B, D, T_spec]
            this_feat = bn(this_feat)
            assert this_feat.shape[-1] == mix_cond.shape[-1], (
                f"Time mismatch between subband feature {this_feat.shape[-1]} and mix_cond "
                f"{mix_cond.shape[-1]}"
            )
            # shape: [B, D, T_spec]
            this_feat = this_feat + mix_cond
            subband_features.append(this_feat)
        subband_feature = torch.stack(subband_features, dim=1)  # shape: [B, nband, D, T_spec]
        assert subband_feature.shape[1] == self.nband
        return subband_feature

    def _reconstruct_waveform(
        self,
        sep_output: torch.Tensor,
        subband_mix_complex: List[torch.Tensor],
        nsample: int,
        mix_wav: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate complex spectrum from masks and reconstruct waveform."""
        bsz = mix_wav.shape[0]
        sep_subband_complex: List[torch.Tensor] = []
        for i, mask_head in enumerate(self.mask_heads):
            # shape: [B, 2, 2, bw, T_spec]
            this_output = mask_head(sep_output[:, i]).view(
                bsz, 2, 2, self.band_width[i], -1
            )
            # shape: [B, 2, bw, T_spec]
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            mask_real = this_mask[:, 0]  # shape: [B, bw, T_spec]
            mask_imag = this_mask[:, 1]  # shape: [B, bw, T_spec]
            mix_real = subband_mix_complex[i].real
            mix_imag = subband_mix_complex[i].imag
            est_real = mix_real * mask_real - mix_imag * mask_imag  # shape: [B, bw, T_spec]
            est_imag = mix_real * mask_imag + mix_imag * mask_real  # shape: [B, bw, T_spec]
            sep_subband_complex.append(torch.complex(est_real, est_imag))

        est_spec = torch.cat(sep_subband_complex, dim=1)  # shape: [B, F, T_spec]
        output = torch.istft(
            est_spec.view(bsz, self.enc_dim, -1),
            n_fft=self.cfg.win,
            hop_length=self.cfg.stride,
            window=torch.hann_window(
                self.cfg.win, dtype=mix_wav.dtype, device=mix_wav.device
            ),
            length=nsample,
        )  # shape: [B, T]
        return output

    def forward(self, mix_wav: torch.Tensor, ref_wav: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        assert mix_wav.ndim == 2, f"Expected mix [B, T], got {tuple(mix_wav.shape)}"
        assert ref_wav.ndim == 2, f"Expected ref [B, T_ref], got {tuple(ref_wav.shape)}"
        assert mix_wav.device == ref_wav.device, "mix/ref device mismatch"
        assert mix_wav.dtype == ref_wav.dtype, "mix/ref dtype mismatch"

        _, nsample = mix_wav.shape
        t_spec, subband_spec_ri, subband_mix_complex = self._extract_stft_subbands(mix_wav)

        # shape: [B, C_ssl, T_ssl, L]
        ssl_mix, _ = self.ssl_frontend(mix_wav, None)  # type: ignore[misc]
        # shape: [B, C_ssl, T_ssl]
        mix_ssl = self.layer_weight(ssl_mix)
        # shape: [B, C_sep, T_ssl]
        mix_cond = self.mix_proj(mix_ssl)
        # shape: [B, C_sep, T_spec]
        mix_cond = F.interpolate(mix_cond, size=t_spec, mode="linear", align_corners=False)

        # shape: [B, 1, spk_emb_dim, 1]
        spk_embedding = self._compute_speaker_embedding(ref_wav)

        subband_feature = self._build_subband_feature_tensor(subband_spec_ri, mix_cond)

        # shape: [B, nband, D, T_spec]
        sep_output = self.separator(subband_feature, spk_embedding)
        return self._reconstruct_waveform(sep_output, subband_mix_complex, nsample, mix_wav)


class WavLMAIEBSRNN(WavLMBSRNN):
    """AIE-enhanced WavLM-conditioned BSRNN for target speaker extraction."""

    def __init__(self, cfg: Optional[WavLMAIEBSRNNConfig] = None, **kwargs) -> None:
        if cfg is None:
            for key in ("joint_training", "multi_task", "speaker_feat"):
                kwargs.pop(key, None)
            cfg = WavLMAIEBSRNNConfig(**kwargs)

        base_cfg = WavLMBSRNNConfig(
            wavlm_name=cfg.wavlm_name,
            wavlm_ckpt=cfg.wavlm_ckpt,
            wavlm_frozen=cfg.wavlm_frozen,
            sr=cfg.sr,
            win=cfg.win,
            stride=cfg.stride,
            feature_dim=cfg.feature_dim,
            num_repeat=cfg.num_repeat,
            use_bidirectional=cfg.use_bidirectional,
            spk_emb_dim=cfg.spk_emb_dim,
            spk_fuse_type=cfg.spk_fuse_type,
            multi_fuse=cfg.multi_fuse,
            use_spk_transform=cfg.use_spk_transform,
        )
        super().__init__(base_cfg)
        self.cfg = cfg
        # Remove legacy mix-conditioning modules from the parent class.
        # WavLMAIEBSRNN replaces `layer_weight -> mix_proj` with AIE, so keeping
        # these trainable modules registered would produce unused parameters in DDP.
        del self.layer_weight
        del self.mix_proj
        assert cfg.aie_enable, "WavLMAIEBSRNN expects aie_enable=True"
        assert cfg.aie_mode in {"fpm", "unet"}, f"Unsupported aie_mode: {cfg.aie_mode}"
        assert cfg.mix_fuse_type in {"add", "concat"}, (
            f"mix_fuse_type must be 'add' or 'concat', got {cfg.mix_fuse_type}"
        )
        self.mix_fuse_type = cfg.mix_fuse_type

        wavlm_cfg = WavLMFrontendConfig(
            name=cfg.wavlm_name,
            path_or_url=cfg.wavlm_ckpt,
            frozen=cfg.wavlm_frozen,
        )
        self.ref_ssl_frontend = None
        if not cfg.use_shared_ssl_for_mix_and_ref:
            self.ref_ssl_frontend = HuggingfaceFrontendWavLM(wavlm_cfg)

        extractor_layers = self.ssl_frontend.upstream_config.get("extractor_conv_layer_config", [])
        assert len(extractor_layers) > 0, "WavLM upstream config must include extractor_conv_layer_config"
        self.cnn_feature_dims = [int(layer_cfg[0]) for layer_cfg in extractor_layers]
        self.cnn_feature_strides = self._build_cnn_feature_strides(extractor_layers)
        self.selected_cnn_levels = self._resolve_cnn_levels(cfg.aie_selected_cnn_levels)
        selected_cnn_dims = [self.cnn_feature_dims[idx] for idx in self.selected_cnn_levels]

        feat_dim = self.ssl_frontend.output_size()
        nb_layer = len(self.ssl_frontend.upstream.encoder.transformer.layers) + 1
        self.aie = None
        self.mix_concat_proj = None
        if cfg.mix_fuse_type == "concat":
            self.mix_concat_proj = nn.Conv1d(
                cfg.feature_dim * 2,
                cfg.feature_dim,
                kernel_size=1,
            )

    @staticmethod
    def _build_cnn_feature_strides(extractor_layers: Sequence[Sequence[int]]) -> List[int]:
        strides: List[int] = []
        cumulative_stride = 1
        for layer_cfg in extractor_layers:
            assert len(layer_cfg) == 3, f"Invalid extractor layer config: {layer_cfg}"
            cumulative_stride *= int(layer_cfg[2])
            strides.append(cumulative_stride)
        return strides

    def _resolve_cnn_levels(self, selected: Union[str, Sequence[int]]) -> List[int]:
        num_levels = len(self.cnn_feature_dims)
        if isinstance(selected, str):
            if selected == "auto":
                final_stride = self.cnn_feature_strides[-1]
                min_stride = max(final_stride // 8, 1)
                levels = [
                    idx for idx, stride in enumerate(self.cnn_feature_strides) if stride >= min_stride
                ]
                if not levels:
                    levels = list(range(max(0, num_levels - 3), num_levels))
                return levels
            levels = [int(part.strip()) for part in selected.split(",") if part.strip()]
        else:
            levels = [int(idx) for idx in selected]
        assert len(levels) > 0, "At least one CNN level must be selected for AIE"
        assert all(0 <= idx < num_levels for idx in levels), (
            f"Invalid CNN level selection {levels}, num_levels={num_levels}"
        )
        return sorted(set(levels))

    def _fuse_mix_condition(
        self,
        subband_spec_ri: List[torch.Tensor],
        mix_cond: torch.Tensor,
    ) -> torch.Tensor:
        assert mix_cond.ndim == 3, f"Expected mix_cond [B, D, T_spec], got {tuple(mix_cond.shape)}"
        bsz = mix_cond.shape[0]
        subband_features: List[torch.Tensor] = []
        for i, bn in enumerate(self.band_bottlenecks):
            bw = self.band_width[i]
            # shape: [B, 2 * bw, T_spec]
            this_feat = subband_spec_ri[i].view(bsz, bw * 2, -1)
            # shape: [B, D, T_spec]
            this_feat = bn(this_feat)
            if self.mix_fuse_type == "add":
                this_feat = this_feat + mix_cond
            else:
                concat_feat = torch.cat([this_feat, mix_cond], dim=1)  # shape: [B, 2D, T_spec]
                this_feat = self.mix_concat_proj(concat_feat)  # shape: [B, D, T_spec]
            subband_features.append(this_feat)
        return torch.stack(subband_features, dim=1)  # shape: [B, nband, D, T_spec]

    def forward(self, mix_wav: torch.Tensor, ref_wav: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        assert mix_wav.ndim == 2, f"Expected mix [B, T], got {tuple(mix_wav.shape)}"
        assert ref_wav.ndim == 2, f"Expected ref [B, T_ref], got {tuple(ref_wav.shape)}"
        assert mix_wav.device == ref_wav.device, "mix/ref device mismatch"
        assert mix_wav.dtype == ref_wav.dtype, "mix/ref dtype mismatch"

        _, nsample = mix_wav.shape
        t_spec, subband_spec_ri, subband_mix_complex = self._extract_stft_subbands(mix_wav)

        mix_features, _ = self.ssl_frontend(
            mix_wav,
            None,
            return_hidden_states=True,
            return_cnn_features=True,
        )
        assert mix_features.transformer_stack is not None, "Transformer features are required for AIE"
        assert mix_features.cnn_features is not None, "CNN features are required for AIE"

        selected_cnn_feats = [mix_features.cnn_features[idx] for idx in self.selected_cnn_levels]
        mix_cond = self.aie(
            cnn_feats=selected_cnn_feats,
            trf_feats=mix_features.transformer_stack,
            target_len=t_spec,
        )  # shape: [B, D, T_spec]
        assert mix_cond.shape[1] == self.cfg.feature_dim, (
            f"AIE output dim mismatch: got {mix_cond.shape[1]}, expected {self.cfg.feature_dim}"
        )

        spk_embedding = self._compute_speaker_embedding(ref_wav)  # shape: [B, 1, E, 1]
        subband_feature = self._fuse_mix_condition(subband_spec_ri, mix_cond)
        sep_output = self.separator(subband_feature, spk_embedding)
        return self._reconstruct_waveform(sep_output, subband_mix_complex, nsample, mix_wav)


def build_wavlm_bsrnn(config: Optional[WavLMBSRNNConfig] = None) -> WavLMBSRNN:
    if config is None:
        config = WavLMBSRNNConfig()
    return WavLMBSRNN(config)


def build_wavlm_aie_bsrnn(
    config: Optional[WavLMAIEBSRNNConfig] = None,
) -> WavLMAIEBSRNN:
    if config is None:
        config = WavLMAIEBSRNNConfig()
    return WavLMAIEBSRNN(config)


if __name__ == "__main__":
    wavlm_ckpt = os.environ.get("WAVLM_CKPT", "").strip()
    if not wavlm_ckpt:
        raise RuntimeError(
            "Set environment variable WAVLM_CKPT to a converted WavLM checkpoint "
            "before running this smoke test."
        )

    # Minimal smoke test (forward + backward) with the real WavLM frontend.
    model = WavLMBSRNN(
        wavlm_ckpt=wavlm_ckpt,
        wavlm_frozen=True,
        feature_dim=64,
        num_repeat=2,
        spk_emb_dim=128,
    )
    x = torch.randn(2, 32000)  # shape: [B, T]
    ref = torch.randn(2, 24000)  # shape: [B, T_ref]
    target = torch.randn(2, 32000)  # shape: [B, T]
    y = model(x, ref)  # shape: [B, T]
    loss = F.l1_loss(y, target)
    loss.backward()
    # A practical sparsity check for one mask head's last conv.
    w = model.mask_heads[0][-1].weight
    sparsity = float((w == 0).sum().item()) / float(w.numel())
    print(f"output_shape={tuple(y.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"mask_head_weight_sparsity={sparsity:.6f}")

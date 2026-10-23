"""DST-style WavLM-conditioned BSRNN for target speaker extraction."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wesep.models.wavlm_bsrnn import FuseSeparation
from wesep.models.wavlm_dynamic_tasnet import (
    WAVLM_FRAME_STRIDE,
    LayerWeightedSum,
    _build_hybrid_attention_mask,
)
from wesep.modules.ssl_backend import SSL_BACKEND_MHFA
from wesep.modules.tasnet.convs import Conv1D
from wesep.modules.wavlm_frontend.model import wav2vec2_model


@dataclass
class WavLMDSTBSRNNConfig:
    wavlm_name: str = "wavlm_base_plus"
    wavlm_ckpt: str = ""
    wavlm_frozen: bool = True

    sr: int = 16000
    sample_rate: Optional[int] = None
    silence_seconds: float = 0.1
    frame_stride: int = WAVLM_FRAME_STRIDE

    win: int = 512
    stride: int = 128
    feature_dim: int = 128
    num_repeat: int = 6
    use_bidirectional: bool = True

    spk_emb_dim: int = 256
    external_spk_dim: int = 192
    spk_fuse_type: str = "FiLM"
    multi_fuse: bool = True

    stage_mode: str = "dst"
    fixed_joint_depth: int = 6
    # 0 = read the separator's speaker summary at the switch layer (legacy behavior).
    # k>0 = read it at joint layer k instead, decoupling readout layer from joint depth
    # (static stage modes only; refine-stage FiLM still uses the switch-layer summary).
    spk_readout_layer: int = 0
    random_depth_choices: Sequence[int] = (2, 4, 6, 8, 10, 12)
    heuristic_cosine_threshold: float = 0.995
    hard_switch_threshold: float = 0.5
    joint_mask_mode: str = "enr_to_mix_blocked"
    use_kv_bias: bool = True
    kv_bias_scale: float = 1.0

    gate_hidden_dim: int = 128
    gate_warmup_steps: int = 10000

    lambda_c: float = 0.05
    lambda_s: float = 0.01
    beta: float = 0.1
    lambda_p: float = 0.1

    use_external_speaker_for_separator: bool = False

    joint_training: bool = True
    multi_task: bool = False
    speaker_feat: bool = False


class WavLMDSTBSRNN(nn.Module):
    def __init__(
        self,
        cfg: Optional[WavLMDSTBSRNNConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if cfg is None:
            for key in ("joint_training", "multi_task", "speaker_feat", "use_spk_transform"):
                kwargs.pop(key, None)
            cfg = WavLMDSTBSRNNConfig(**kwargs)
        self.cfg = cfg
        self._validate_config()

        self.sample_rate = cfg.sample_rate or cfg.sr
        self.enc_dim = cfg.win // 2 + 1
        self.band_width = self._build_bandwidth(cfg.sr, self.enc_dim)
        self.nband = len(self.band_width)

        self.upstream, self.feat_dim, self.num_layers = self._build_upstream()
        self._configure_upstream_training()
        self.mix_layer_weight = LayerWeightedSum(self.num_layers)

        self.per_layer_mhfa = SSL_BACKEND_MHFA(
            head_nb=8,
            feat_dim=self.feat_dim,
            compression_dim=128,
            embed_dim=cfg.spk_emb_dim,
            nb_layer=1,
            feature_grad_mult=1.0,
        )
        self.gate_mix_proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.gate_speaker_proj = nn.Linear(cfg.spk_emb_dim, self.feat_dim)
        self.gate_energy_query = nn.Linear(cfg.spk_emb_dim, self.feat_dim)
        self.gate_energy_key = nn.Linear(self.feat_dim, self.feat_dim)
        self.gate_energy_value = nn.Linear(self.feat_dim, self.feat_dim)
        self.stage_gate = nn.Sequential(
            nn.Linear(self.feat_dim + cfg.spk_emb_dim + 2, cfg.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden_dim, 1),
        )

        self.refine_key_proj = nn.Linear(cfg.spk_emb_dim, self.feat_dim)
        self.refine_value_proj = nn.Linear(cfg.spk_emb_dim, self.feat_dim)
        self.mix_proj = Conv1D(self.feat_dim, cfg.feature_dim, kernel_size=1)
        self.external_spk_proj = nn.Linear(cfg.external_spk_dim, cfg.spk_emb_dim)

        eps = torch.finfo(torch.float32).eps
        self.band_bottlenecks = nn.ModuleList()
        for width in self.band_width:
            self.band_bottlenecks.append(
                nn.Sequential(
                    nn.GroupNorm(1, width * 2, eps),
                    nn.Conv1d(width * 2, cfg.feature_dim, kernel_size=1),
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
        for width in self.band_width:
            self.mask_heads.append(
                nn.Sequential(
                    nn.GroupNorm(1, cfg.feature_dim, eps),
                    nn.Conv1d(cfg.feature_dim, cfg.feature_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(cfg.feature_dim * 4, cfg.feature_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(cfg.feature_dim * 4, width * 4, kernel_size=1),
                )
            )

        self.register_buffer("stft_window", torch.hann_window(cfg.win), persistent=False)
        self.current_step = 0
        self._last_dynamic_stats: dict[str, torch.Tensor] = {}

    def _validate_config(self) -> None:
        valid_stage_modes = {
            "late_ssl",
            "full_prompt",
            "fixed_stage",
            "random_stage",
            "heuristic_stage",
            "dst",
        }
        if self.cfg.stage_mode not in valid_stage_modes:
            raise ValueError(
                f"stage_mode must be one of {sorted(valid_stage_modes)}, got {self.cfg.stage_mode}"
            )
        if self.cfg.joint_mask_mode not in {"full", "enr_to_mix_blocked", "fully_blocked"}:
            raise ValueError(
                "joint_mask_mode must be one of ['full', 'enr_to_mix_blocked', 'fully_blocked']"
            )
        if not self.cfg.wavlm_ckpt:
            raise ValueError("wavlm_ckpt must be provided")
        if self.cfg.silence_seconds < 0.0:
            raise ValueError("silence_seconds must be >= 0")
        if self.cfg.kv_bias_scale < 0.0:
            raise ValueError("kv_bias_scale must be >= 0")
        if self.cfg.feature_dim <= 0 or self.cfg.spk_emb_dim <= 0:
            raise ValueError("feature_dim and spk_emb_dim must be > 0")
        if self.cfg.spk_readout_layer < 0:
            raise ValueError("spk_readout_layer must be >= 0")
        if self.cfg.spk_readout_layer > 0:
            if self.cfg.stage_mode == "dst":
                raise ValueError("spk_readout_layer requires a static stage_mode")
            if (
                self.cfg.stage_mode == "fixed_stage"
                and self.cfg.spk_readout_layer > int(self.cfg.fixed_joint_depth)
            ):
                raise ValueError("spk_readout_layer must be <= fixed_joint_depth")

    def set_current_step(self, step: int) -> None:
        self.current_step = int(step)

    def get_last_dynamic_stats(
        self,
        *,
        detach: bool = True,
        to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        stats = dict(self._last_dynamic_stats)
        if detach:
            stats = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in stats.items()
            }
        if to_cpu:
            stats = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in stats.items()
            }
        return stats

    def _build_upstream(self) -> tuple[nn.Module, int, int]:
        state = torch.load(self.cfg.wavlm_ckpt, map_location="cpu")
        if "config" not in state or "state_dict" not in state:
            raise ValueError("Converted WavLM checkpoint must contain 'config' and 'state_dict'")
        upstream_config = dict(state["config"])
        upstream_config.update(
            dict(
                extractor_prune_conv_channels=False,
                encoder_prune_attention_heads=False,
                encoder_prune_attention_layer=False,
                encoder_prune_feed_forward_intermediate=False,
                encoder_prune_feed_forward_layer=False,
            )
        )
        upstream = wav2vec2_model(**upstream_config)
        upstream.load_state_dict(state["state_dict"], strict=False)
        feat_dim = int(upstream_config["encoder_embed_dim"])
        num_layers = len(upstream.encoder.transformer.layers)
        return upstream, feat_dim, num_layers

    def _configure_upstream_training(self) -> None:
        for parameter in self.upstream.parameters():
            parameter.requires_grad_(not self.cfg.wavlm_frozen)
        self._has_trainable_upstream = any(
            parameter.requires_grad for parameter in self.upstream.parameters()
        )

    def _upstream_context(self):
        if self._has_trainable_upstream:
            return contextlib.nullcontext()
        return torch.no_grad()

    @staticmethod
    def _build_bandwidth(sr: int, enc_dim: int) -> list[int]:
        b100 = int(np.floor(100 / (sr / 2.0) * enc_dim))
        b200 = int(np.floor(200 / (sr / 2.0) * enc_dim))
        b500 = int(np.floor(500 / (sr / 2.0) * enc_dim))
        b2k = int(np.floor(2000 / (sr / 2.0) * enc_dim))
        widths = [b100] * 15 + [b200] * 10 + [b500] * 5 + [b2k]
        widths.append(enc_dim - int(np.sum(widths)))
        return widths

    def _default_mix_start(self, enroll_len: int) -> int:
        return enroll_len + int(self.cfg.silence_seconds * self.sample_rate)

    def _get_frame_indices(
        self,
        total_frames: int,
        enroll_len: int,
        mix_start: int,
    ) -> tuple[int, int]:
        enroll_end_frame = min(enroll_len // self.cfg.frame_stride, total_frames)
        mix_start_frame = min(mix_start // self.cfg.frame_stride, total_frames)
        return enroll_end_frame, mix_start_frame

    def _build_joint_attention_mask(
        self,
        total_frames: int,
        enroll_end_frame: int,
        mix_start_frame: int,
        pad_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        mask: Optional[torch.Tensor]
        if self.cfg.joint_mask_mode == "full":
            mask = None
        elif self.cfg.joint_mask_mode == "enr_to_mix_blocked":
            mask = _build_hybrid_attention_mask(
                total_frames,
                enroll_end_frame,
                mix_start_frame,
                device,
                dtype,
                asymmetric=True,
            )
        else:
            mask = _build_hybrid_attention_mask(
                total_frames,
                enroll_end_frame,
                mix_start_frame,
                device,
                dtype,
                asymmetric=False,
            )
        if pad_mask is not None:
            mask = pad_mask if mask is None else (mask + pad_mask)
        return mask

    def _summarize_single_layer(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [B, T, C], got {tuple(hidden.shape)}")
        stacked = hidden.transpose(1, 2).contiguous().unsqueeze(-1)
        return self.per_layer_mhfa(stacked)

    def _extract_enroll_only_summary(self, enroll_wav: torch.Tensor) -> torch.Tensor:
        with self._upstream_context():
            feat, _ = self.upstream.feature_extractor(enroll_wav, None)
        x, pad_mask = self.upstream.encoder._preprocess(feat, None)
        x = self.upstream.encoder.transformer._preprocess(x)
        position_bias = None
        speaker_summary = None
        for layer in self.upstream.encoder.transformer.layers:
            x, position_bias = layer(x, pad_mask, position_bias=position_bias)
            speaker_summary = self._summarize_single_layer(x)
        if speaker_summary is None:
            raise RuntimeError("Failed to compute enrollment summary")
        return speaker_summary

    def _compute_alignment_statistics(
        self,
        mix_hidden: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mix_pool = mix_hidden.mean(dim=1)
        cosine = F.cosine_similarity(
            self.gate_mix_proj(mix_pool),
            self.gate_speaker_proj(speaker_emb),
            dim=-1,
        )

        query = self.gate_energy_query(speaker_emb).unsqueeze(1)
        key = self.gate_energy_key(mix_hidden)
        logits = torch.sum(query * key, dim=-1) / np.sqrt(float(self.feat_dim))
        weights = torch.softmax(logits, dim=-1)
        value = self.gate_energy_value(mix_hidden)
        energy = (weights.unsqueeze(-1) * value.square()).sum(dim=(1, 2))
        energy = energy / float(value.size(-1))
        return cosine, energy

    def _compute_stage_gate(
        self,
        mix_hidden: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mix_pool = mix_hidden.mean(dim=1)
        cosine, energy = self._compute_alignment_statistics(mix_hidden, speaker_emb)
        gate_input = torch.cat(
            [mix_pool, speaker_emb, cosine.unsqueeze(-1), energy.unsqueeze(-1)],
            dim=-1,
        )
        gate_prob = torch.sigmoid(self.stage_gate(gate_input))
        return gate_prob, cosine, energy

    def _run_refine_layer(
        self,
        layer: nn.Module,
        mix_hidden: torch.Tensor,
        speaker_emb: torch.Tensor,
        position_bias: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        speaker_key = self.refine_key_proj(speaker_emb)
        speaker_value = self.refine_value_proj(speaker_emb)

        attn_bias = torch.zeros(
            mix_hidden.size(0),
            1,
            1,
            mix_hidden.size(1),
            device=mix_hidden.device,
            dtype=mix_hidden.dtype,
        )
        if self.cfg.use_kv_bias:
            attn_bias = (
                self.cfg.kv_bias_scale
                * F.cosine_similarity(
                    F.normalize(mix_hidden, dim=-1),
                    F.normalize(speaker_key.unsqueeze(1), dim=-1),
                    dim=-1,
                ).unsqueeze(1).unsqueeze(2)
            )
            mix_hidden = mix_hidden + self.cfg.kv_bias_scale * speaker_value.unsqueeze(1)
        next_hidden, position_bias = layer(
            mix_hidden,
            attn_bias,
            position_bias=position_bias,
        )
        return next_hidden, position_bias, attn_bias

    def _prepare_separator_embedding(
        self,
        internal_spk: Optional[torch.Tensor],
        external_spk_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.cfg.use_external_speaker_for_separator:
            if external_spk_emb is None:
                raise ValueError("external_spk_emb must be provided when use_external_speaker_for_separator=true")
            return self.external_spk_proj(external_spk_emb)
        if internal_spk is None:
            raise ValueError("internal speaker embedding is required for separator conditioning")
        return internal_spk

    def _resize_1d_linear(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(-1) == target_len:
            return x
        if target_len <= 1:
            return x[..., :target_len]
        source_len = x.size(-1)
        positions = torch.linspace(
            0,
            source_len - 1,
            target_len,
            device=x.device,
            dtype=x.dtype,
        )
        left = positions.floor().long()
        right = torch.clamp(left + 1, max=source_len - 1)
        weight = (positions - left.to(dtype=x.dtype)).view(1, 1, -1)
        left_value = torch.index_select(x, dim=-1, index=left)
        right_value = torch.index_select(x, dim=-1, index=right)
        return left_value * (1.0 - weight) + right_value * weight

    def _select_static_joint_depth(self, key: str) -> int:
        if self.cfg.stage_mode == "full_prompt":
            return self.num_layers
        if self.cfg.stage_mode == "late_ssl":
            return 0
        if self.cfg.stage_mode == "fixed_stage":
            return min(max(int(self.cfg.fixed_joint_depth), 0), self.num_layers)
        if self.cfg.stage_mode == "random_stage":
            choices = [min(max(int(item), 0), self.num_layers) for item in self.cfg.random_depth_choices]
            choices = [item for item in choices if item <= self.num_layers]
            if not choices:
                return self.num_layers
            return int(random.choice(choices))
        return self.num_layers

    def _estimate_attention_stats(
        self,
        *,
        batch_size: int,
        total_frames: int,
        mix_frames: int,
        joint_depth: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        joint_cost = float(total_frames * total_frames)
        refine_cost = float(mix_frames * mix_frames)
        flops = joint_depth * joint_cost + (self.num_layers - joint_depth) * refine_cost
        peak = max(joint_cost, refine_cost) * bytes_per_element * batch_size
        peak_tensor = torch.full_like(flops, float(peak))
        return flops, peak_tensor

    def _forward_ssl_soft(
        self,
        x0: torch.Tensor,
        joint_mask: Optional[torch.Tensor],
        enroll_end_frame: int,
        mix_start_frame: int,
        *,
        enroll_wav: torch.Tensor,
        external_spk_emb: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = x0.size(0)
        mix_hidden = x0[:, mix_start_frame:, :].contiguous()
        x_joint = x0
        joint_position_bias = None
        refine_position_bias = None

        mix_outputs: list[torch.Tensor] = []
        gate_probs: list[torch.Tensor] = []
        gate_cosines: list[torch.Tensor] = []
        gate_energies: list[torch.Tensor] = []
        monotonic_probs: list[torch.Tensor] = []

        expected_joint_depth = torch.zeros(batch_size, device=x0.device)
        current_monotonic = torch.ones(batch_size, 1, device=x0.device)
        speaker_prev: Optional[torch.Tensor] = None
        speaker_switch: Optional[torch.Tensor] = None
        heuristic_depth = self.num_layers
        fixed_depth = self._select_static_joint_depth(self.cfg.stage_mode)

        if self.cfg.stage_mode == "late_ssl":
            position_bias = None
            for layer in self.upstream.encoder.transformer.layers:
                mix_hidden, position_bias = layer(
                    mix_hidden,
                    None,
                    position_bias=position_bias,
                )
                mix_outputs.append(mix_hidden)
            speaker_for_separator = self._prepare_separator_embedding(
                self._extract_enroll_only_summary(enroll_wav),
                external_spk_emb,
            )
            joint_depth = torch.zeros(batch_size, device=x0.device)
        elif self.cfg.stage_mode == "dst":
            hard_lstar = torch.full(
                (batch_size,),
                float(self.num_layers),
                device=x0.device,
            )
            for layer_idx, layer in enumerate(self.upstream.encoder.transformer.layers):
                x_joint_next, joint_position_bias = layer(
                    x_joint,
                    joint_mask,
                    position_bias=joint_position_bias,
                )
                joint_mix = x_joint_next[:, mix_start_frame:, :].contiguous()
                speaker_now = self._summarize_single_layer(
                    x_joint_next[:, :enroll_end_frame, :].contiguous()
                )
                speaker_for_refine = speaker_prev if speaker_prev is not None else speaker_now
                refine_next, refine_position_bias, _ = self._run_refine_layer(
                    layer,
                    mix_hidden,
                    speaker_for_refine,
                    refine_position_bias,
                )
                gate_prob, gate_cos, gate_energy = self._compute_stage_gate(joint_mix, speaker_now)
                if self.training and self.current_step < self.cfg.gate_warmup_steps:
                    gate_prob = torch.zeros_like(gate_prob)
                gate_probs.append(gate_prob.squeeze(-1))
                gate_cosines.append(gate_cos)
                gate_energies.append(gate_energy)
                monotonic_probs.append(current_monotonic.squeeze(-1))
                expected_joint_depth += current_monotonic.squeeze(-1)
                mix_hidden = (
                    current_monotonic.unsqueeze(-1) * joint_mix
                    + (1.0 - current_monotonic).unsqueeze(-1) * refine_next
                )
                current_monotonic = current_monotonic * (1.0 - gate_prob)
                speaker_prev = speaker_now
                x_joint = x_joint_next
                mix_outputs.append(mix_hidden)

            speaker_switch = speaker_prev
            speaker_for_separator = self._prepare_separator_embedding(
                speaker_switch,
                external_spk_emb,
            )
            if gate_probs:
                gate_stack = torch.stack(gate_probs, dim=1)
                above = gate_stack >= float(self.cfg.hard_switch_threshold)
                for batch_idx in range(batch_size):
                    indices = torch.nonzero(above[batch_idx], as_tuple=False)
                    if indices.numel() > 0:
                        hard_lstar[batch_idx] = float(indices[0, 0].item() + 1)
            joint_depth = expected_joint_depth
        else:
            speaker_last: Optional[torch.Tensor] = None
            speaker_readout: Optional[torch.Tensor] = None
            position_bias = None
            depth = fixed_depth
            for layer_idx, layer in enumerate(self.upstream.encoder.transformer.layers):
                if layer_idx < depth:
                    x_joint, position_bias = layer(
                        x_joint,
                        joint_mask,
                        position_bias=position_bias,
                    )
                    mix_hidden = x_joint[:, mix_start_frame:, :].contiguous()
                    speaker_now = self._summarize_single_layer(
                        x_joint[:, :enroll_end_frame, :].contiguous()
                    )
                    speaker_last = speaker_now
                    if self.cfg.spk_readout_layer and layer_idx + 1 == int(self.cfg.spk_readout_layer):
                        speaker_readout = speaker_now
                    if (
                        self.cfg.stage_mode == "heuristic_stage"
                        and layer_idx + 1 < self.num_layers
                        and speaker_prev is not None
                    ):
                        cosine = F.cosine_similarity(speaker_now, speaker_prev, dim=-1)
                        if float(cosine.mean().item()) >= self.cfg.heuristic_cosine_threshold:
                            heuristic_depth = layer_idx + 1
                            depth = heuristic_depth
                    speaker_prev = speaker_now
                else:
                    if speaker_last is None:
                        raise RuntimeError("speaker_last must be available before entering refine stage")
                    mix_hidden, refine_position_bias, _ = self._run_refine_layer(
                        layer,
                        mix_hidden,
                        speaker_last,
                        refine_position_bias,
                    )
                mix_outputs.append(mix_hidden)

            if self.cfg.spk_readout_layer:
                if speaker_readout is None:
                    raise RuntimeError(
                        f"spk_readout_layer={self.cfg.spk_readout_layer} was never reached "
                        f"(effective joint depth {depth}); no readout summary captured"
                    )
                separator_summary = speaker_readout
            else:
                separator_summary = speaker_last
            speaker_for_separator = self._prepare_separator_embedding(separator_summary, external_spk_emb)
            actual_depth = heuristic_depth if self.cfg.stage_mode == "heuristic_stage" else depth
            joint_depth = torch.full(
                (batch_size,),
                float(actual_depth),
                device=x0.device,
            )

        if not mix_outputs:
            raise RuntimeError("No SSL mixture outputs were produced")

        total_frames = x0.size(1)
        mix_frames = mix_outputs[-1].size(1)
        flops, peak_mem = self._estimate_attention_stats(
            batch_size=batch_size,
            total_frames=total_frames,
            mix_frames=mix_frames,
            joint_depth=joint_depth,
            dtype=x0.dtype,
        )
        stats: dict[str, torch.Tensor] = {
            "expected_joint_depth": joint_depth,
            "attn_flops": flops,
            "peak_attn_mem_bytes": peak_mem,
            "mix_frames": torch.full((batch_size,), float(mix_frames), device=x0.device),
            "total_frames": torch.full((batch_size,), float(total_frames), device=x0.device),
            "enroll_end_frame": torch.full(
                (batch_size,), float(enroll_end_frame), device=x0.device
            ),
        }
        if gate_probs:
            stats["gate_probs"] = torch.stack(gate_probs, dim=1)
            stats["monotonic_probs"] = torch.stack(monotonic_probs, dim=1)
            stats["gate_cosines"] = torch.stack(gate_cosines, dim=1)
            stats["gate_energies"] = torch.stack(gate_energies, dim=1)
        else:
            stats["gate_probs"] = torch.zeros(batch_size, self.num_layers, device=x0.device)
            stats["monotonic_probs"] = torch.zeros(batch_size, self.num_layers, device=x0.device)
        if self.cfg.stage_mode == "dst":
            stats["hard_lstar"] = hard_lstar
        else:
            stats["hard_lstar"] = joint_depth
        return mix_outputs, speaker_for_separator, stats

    def _forward_ssl_hard_single(
        self,
        x0: torch.Tensor,
        joint_mask: Optional[torch.Tensor],
        enroll_end_frame: int,
        mix_start_frame: int,
        *,
        enroll_wav: torch.Tensor,
        external_spk_emb: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        mix_hidden = x0[:, mix_start_frame:, :].contiguous()
        x_joint = x0
        joint_position_bias = None
        refine_position_bias = None
        mix_outputs: list[torch.Tensor] = []
        gate_probs: list[torch.Tensor] = []
        gate_cosines: list[torch.Tensor] = []
        gate_energies: list[torch.Tensor] = []

        switched = False
        joint_depth = 0
        speaker_switch: Optional[torch.Tensor] = None

        for layer_idx, layer in enumerate(self.upstream.encoder.transformer.layers):
            if not switched:
                x_joint, joint_position_bias = layer(
                    x_joint,
                    joint_mask,
                    position_bias=joint_position_bias,
                )
                mix_hidden = x_joint[:, mix_start_frame:, :].contiguous()
                speaker_now = self._summarize_single_layer(
                    x_joint[:, :enroll_end_frame, :].contiguous()
                )
                gate_prob, gate_cos, gate_energy = self._compute_stage_gate(mix_hidden, speaker_now)
                gate_probs.append(gate_prob.squeeze(-1))
                gate_cosines.append(gate_cos)
                gate_energies.append(gate_energy)
                mix_outputs.append(mix_hidden)
                joint_depth = layer_idx + 1
                speaker_switch = speaker_now
                if (
                    float(gate_prob.squeeze().item()) >= self.cfg.hard_switch_threshold
                    and layer_idx + 1 < self.num_layers
                ):
                    switched = True
            else:
                if speaker_switch is None:
                    raise RuntimeError("speaker_switch must be available for refine mode")
                mix_hidden, refine_position_bias, _ = self._run_refine_layer(
                    layer,
                    mix_hidden,
                    speaker_switch,
                    refine_position_bias,
                )
                mix_outputs.append(mix_hidden)

        if speaker_switch is None:
            speaker_switch = self._extract_enroll_only_summary(enroll_wav)
        speaker_for_separator = self._prepare_separator_embedding(
            speaker_switch,
            external_spk_emb,
        )
        def pad_layer_stats(values: list[torch.Tensor]) -> torch.Tensor:
            if not values:
                return torch.zeros(1, self.num_layers, device=x0.device)
            stacked = torch.stack(values, dim=1)
            if stacked.size(1) >= self.num_layers:
                return stacked[:, : self.num_layers]
            pad = torch.zeros(
                stacked.size(0),
                self.num_layers - stacked.size(1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
            return torch.cat([stacked, pad], dim=1)

        total_frames = x0.size(1)
        mix_frames = mix_outputs[-1].size(1)
        joint_depth_tensor = torch.tensor([float(joint_depth)], device=x0.device)
        flops, peak_mem = self._estimate_attention_stats(
            batch_size=1,
            total_frames=total_frames,
            mix_frames=mix_frames,
            joint_depth=joint_depth_tensor,
            dtype=x0.dtype,
        )
        stats = {
            "gate_probs": pad_layer_stats(gate_probs),
            "gate_cosines": pad_layer_stats(gate_cosines),
            "gate_energies": pad_layer_stats(gate_energies),
            "monotonic_probs": torch.zeros(1, self.num_layers, device=x0.device),
            "expected_joint_depth": joint_depth_tensor,
            "hard_lstar": joint_depth_tensor.clone(),
            "attn_flops": flops,
            "peak_attn_mem_bytes": peak_mem,
            "mix_frames": torch.tensor([float(mix_frames)], device=x0.device),
            "total_frames": torch.tensor([float(total_frames)], device=x0.device),
            "enroll_end_frame": torch.tensor([float(enroll_end_frame)], device=x0.device),
            "hard_eval_no_enroll_after_switch": torch.tensor([1.0], device=x0.device),
        }
        return mix_outputs, speaker_for_separator, stats

    def _stack_stats(self, stats_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        merged: dict[str, list[torch.Tensor]] = {}
        for stats in stats_list:
            for key, value in stats.items():
                merged.setdefault(key, []).append(value)
        stacked: dict[str, torch.Tensor] = {}
        for key, values in merged.items():
            try:
                stacked[key] = torch.cat(values, dim=0)
                continue
            except RuntimeError as exc:
                ranks = {value.dim() for value in values}
                if len(ranks) != 1 or 0 in ranks:
                    shapes = [tuple(value.shape) for value in values]
                    raise RuntimeError(f"Cannot stack DST stat {key!r} with shapes {shapes}") from exc

                target_shape = list(values[0].shape)
                for value in values[1:]:
                    for dim in range(1, value.dim()):
                        target_shape[dim] = max(target_shape[dim], value.size(dim))

                padded_values = []
                for value in values:
                    pad_width: list[int] = []
                    for dim in range(value.dim() - 1, 0, -1):
                        pad_width.extend([0, target_shape[dim] - value.size(dim)])
                    padded_values.append(F.pad(value, pad_width) if any(pad_width) else value)
                stacked[key] = torch.cat(padded_values, dim=0)
        return stacked

    def _run_ssl_branch(
        self,
        mix_wav: torch.Tensor,
        enroll_wav: torch.Tensor,
        external_spk_emb: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        silence = torch.zeros(
            mix_wav.size(0),
            int(self.cfg.silence_seconds * self.sample_rate),
            dtype=mix_wav.dtype,
            device=mix_wav.device,
        )
        wav = torch.cat([enroll_wav, silence, mix_wav], dim=-1)
        with self._upstream_context():
            feat, _ = self.upstream.feature_extractor(wav, None)
        total_frames = feat.size(1)
        enroll_end_frame, mix_start_frame = self._get_frame_indices(
            total_frames,
            enroll_wav.size(1),
            self._default_mix_start(enroll_wav.size(1)),
        )
        x0, pad_mask = self.upstream.encoder._preprocess(feat, None)
        x0 = self.upstream.encoder.transformer._preprocess(x0)
        joint_mask = self._build_joint_attention_mask(
            total_frames,
            enroll_end_frame,
            mix_start_frame,
            pad_mask,
            x0.device,
            x0.dtype,
        )

        if (not self.training) and self.cfg.stage_mode == "dst" and mix_wav.size(0) > 1:
            all_outputs = []
            all_spk = []
            all_stats = []
            for batch_idx in range(mix_wav.size(0)):
                sample_outputs, sample_spk, sample_stats = self._forward_ssl_hard_single(
                    x0[batch_idx : batch_idx + 1],
                    joint_mask,
                    enroll_end_frame,
                    mix_start_frame,
                    enroll_wav=enroll_wav[batch_idx : batch_idx + 1],
                    external_spk_emb=external_spk_emb[batch_idx : batch_idx + 1]
                    if external_spk_emb is not None
                    else None,
                )
                all_outputs.append(sample_outputs)
                all_spk.append(sample_spk)
                all_stats.append(sample_stats)
            mix_outputs = [
                torch.cat([sample_outputs[layer_idx] for sample_outputs in all_outputs], dim=0)
                for layer_idx in range(self.num_layers)
            ]
            speaker_for_separator = torch.cat(all_spk, dim=0)
            stats = self._stack_stats(all_stats)
            return mix_outputs, speaker_for_separator, stats

        if (not self.training) and self.cfg.stage_mode == "dst":
            return self._forward_ssl_hard_single(
                x0,
                joint_mask,
                enroll_end_frame,
                mix_start_frame,
                enroll_wav=enroll_wav,
                external_spk_emb=external_spk_emb,
            )
        return self._forward_ssl_soft(
            x0,
            joint_mask,
            enroll_end_frame,
            mix_start_frame,
            enroll_wav=enroll_wav,
            external_spk_emb=external_spk_emb,
        )

    def _separate(
        self,
        mix_wav: torch.Tensor,
        mix_outputs: list[torch.Tensor],
        speaker_for_separator: torch.Tensor,
    ) -> torch.Tensor:
        aggregate_mix = self.mix_layer_weight(mix_outputs).transpose(1, 2).contiguous()
        spec = torch.stft(
            mix_wav,
            n_fft=self.cfg.win,
            hop_length=self.cfg.stride,
            window=self.stft_window.to(device=mix_wav.device, dtype=mix_wav.dtype),
            return_complex=True,
        )
        time_steps = spec.size(-1)
        cond = self.mix_proj(aggregate_mix)
        cond = self._resize_1d_linear(cond, time_steps)

        spec_ri = torch.stack([spec.real, spec.imag], dim=1)
        subband_spec_ri = []
        subband_mix_complex = []
        band_index = 0
        for width in self.band_width:
            subband_spec_ri.append(spec_ri[:, :, band_index : band_index + width].contiguous())
            subband_mix_complex.append(spec[:, band_index : band_index + width])
            band_index += width

        subband_features = []
        for band_id, bottleneck in enumerate(self.band_bottlenecks):
            width = self.band_width[band_id]
            feature = subband_spec_ri[band_id].view(mix_wav.size(0), width * 2, -1)
            feature = bottleneck(feature) + cond
            subband_features.append(feature)
        subband_feature = torch.stack(subband_features, dim=1)
        spk_embedding = speaker_for_separator.unsqueeze(1).unsqueeze(3)
        sep_output = self.separator(subband_feature, spk_embedding)

        sep_subband_complex = []
        for band_id, mask_head in enumerate(self.mask_heads):
            this_output = mask_head(sep_output[:, band_id]).view(
                mix_wav.size(0),
                2,
                2,
                self.band_width[band_id],
                -1,
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            mask_real = this_mask[:, 0]
            mask_imag = this_mask[:, 1]
            mix_real = subband_mix_complex[band_id].real
            mix_imag = subband_mix_complex[band_id].imag
            est_real = mix_real * mask_real - mix_imag * mask_imag
            est_imag = mix_real * mask_imag + mix_imag * mask_real
            sep_subband_complex.append(torch.complex(est_real, est_imag))

        est_spec = torch.cat(sep_subband_complex, dim=1)
        return torch.istft(
            est_spec.view(mix_wav.size(0), self.enc_dim, -1),
            n_fft=self.cfg.win,
            hop_length=self.cfg.stride,
            window=self.stft_window.to(device=mix_wav.device, dtype=mix_wav.dtype),
            length=mix_wav.size(-1),
        )

    def forward(
        self,
        mix_wav: torch.Tensor,
        enroll_wav: torch.Tensor,
        external_spk_emb: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mix_wav.ndim != 2 or enroll_wav.ndim != 2:
            raise ValueError(
                f"Expected 2-D tensors [B, T], got mix={tuple(mix_wav.shape)}, enroll={tuple(enroll_wav.shape)}"
            )
        ssl_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if mix_wav.is_cuda
            else contextlib.nullcontext()
        )
        ssl_external_spk_emb = (
            external_spk_emb.float()
            if external_spk_emb is not None and torch.is_floating_point(external_spk_emb)
            else external_spk_emb
        )
        with ssl_context:
            mix_outputs, speaker_for_separator, stats = self._run_ssl_branch(
                mix_wav.float(),
                enroll_wav.float(),
                ssl_external_spk_emb,
            )
        separated = self._separate(mix_wav, mix_outputs, speaker_for_separator)
        stats["mean_gate_prob"] = stats["gate_probs"].mean(dim=1)
        self._last_dynamic_stats = stats
        if return_embedding:
            return separated, speaker_for_separator
        return separated


def build_wavlm_dst_bsrnn(
    config: Optional[WavLMDSTBSRNNConfig] = None,
) -> WavLMDSTBSRNN:
    if config is None:
        config = WavLMDSTBSRNNConfig()
    return WavLMDSTBSRNN(config)

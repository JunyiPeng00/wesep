"""WavLM frontend for wesep (no pruning, layer-wise outputs).

This module provides a high-level wrapper that loads a locally converted
WavLM checkpoint (config + state_dict) and exposes layer-wise hidden states
with a fixed frame rate (~20 ms), matching the behavior of the original
`wespeaker_hubert` `HuggingfaceFrontend` but without any pruning control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import contextlib
import logging

import torch
import torch.nn as nn

from .model import wav2vec2_model

logger = logging.getLogger(__name__)


@dataclass
class WavLMFrontendConfig:
    """Configuration for HuggingfaceFrontendWavLM.

    Attributes:
        name:
            Model name string (e.g. "wavlm_base"), used only for dimension
            sanity checks and logging.
        path_or_url:
            Path to a locally converted checkpoint containing a dict with
            keys ``\"config\"`` (kwargs for `wav2vec2_model`) and
            ``\"state_dict\"`` (model weights).
        frozen:
            If True, disable gradient on the upstream WavLM.
    """

    name: str
    path_or_url: str
    frozen: bool = False


class HuggingfaceFrontendWavLM(nn.Module):
    """Speech SSL frontend based on a locally converted WavLM checkpoint.

    The forward method returns a 4-D tensor of layer-wise hidden states with
    shape ``[B, C, T, L]`` where:

    - ``B``: batch size
    - ``C``: feature dimension (e.g. 768 for WavLM-Base)
    - ``T``: time frames at ~20 ms resolution
    - ``L``: number of layers (including the pre-Transformer representation)
    """

    def __init__(self, cfg: WavLMFrontendConfig) -> None:
        super().__init__()
        self.upstream_name = cfg.name.lower()
        self.frozen = cfg.frozen

        self.upstream, self.upstream_config = self._build_upstream(cfg.path_or_url)

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                # Mask embedding is not useful for separation and can stay frozen
                if "mask_emb" in name:
                    param.requires_grad_(False)

    def _build_upstream(self, upstream_ckpt: str):
        """Build the upstream WavLM model from a converted checkpoint."""
        state = torch.load(upstream_ckpt, map_location="cpu")
        upstream_config = state["config"]
        # Ensure all pruning-related flags are disabled; wesep does not use them.
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
        load_result = upstream.load_state_dict(state["state_dict"], strict=False)
        logger.info(
            "Loaded pretrained WavLM ckpt: missing %s, unexpected %s",
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
        return upstream, upstream_config

    def get_num_params(self) -> int:
        """Return the number of parameters of the upstream model."""
        return self.upstream.get_num_params()

    def output_size(self) -> int:
        """Return feature dimension of the WavLM encoder output."""
        if "large" in self.upstream_name or "xlsr" in self.upstream_name:
            return 1024
        if "base" in self.upstream_name:
            return 768
        if self.upstream_name in {"xls_r_300m", "xls_r_1b"}:
            return 1024
        if self.upstream_name == "xls_r_2b":
            return 1920
        raise ValueError(f"Unknown model size for: {self.upstream_name}")

    def forward(
        self,
        input_wav: torch.Tensor,
        input_lengths: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # type: ignore[override]
        """Extract layer-wise representations from input waveform.

        Args:
            input_wav:
                Tensor of shape ``[B, T]`` with 16 kHz mono audio.
            input_lengths:
                Optional valid lengths (in samples) for each element in batch.

        Returns:
            layer_reps:
                Tensor of shape ``[B, C, T_frames, L]``.
            out_lengths:
                Optional lengths in frame domain.
        """
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # Do not truncate; pass full waveforms and original lengths.
            ssl_hiddens, out_lengths = self.upstream.extract_features(input_wav, input_lengths)

        # ssl_hiddens: list of [B, T, C] for each layer
        layer_reps = torch.stack(list(ssl_hiddens))  # [L, B, T, C]
        layer_reps = layer_reps.permute(1, 3, 2, 0)  # [B, C, T, L]
        return layer_reps, out_lengths


"""Speech SSL models (WavLM/Wav2Vec2-style) used inside wesep.

This is a lightly adapted copy of the wav2vec2/WavLM wrapper used in
`wespeaker_hubert`, with pruning APIs kept only for checkpoint compatibility.
The wesep separation models do not expose or rely on pruning at runtime.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from . import components


class Wav2Vec2Model(Module):
    """Wrapper around feature extractor + Transformer encoder.

    This class mimics torchaudio's ``Wav2Vec2Model`` interface and exposes:

    - ``forward`` for end-to-end logits (not used in wesep).
    - ``extract_features`` to obtain intermediate layer representations.
    - ``get_num_params`` to estimate parameter count.
    """

    def __init__(
        self,
        normalize_waveform: bool,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
        feature_grad_mult: float = 0.1,
    ) -> None:
        super().__init__()
        self.normalize_waveform = normalize_waveform
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux
        self.feature_grad_mult = feature_grad_mult

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract intermediate layer features from raw waveforms.

        Args:
            waveforms:
                Audio tensor of shape ``(batch, frames)``.
            lengths:
                Optional valid lengths tensor of shape ``(batch,)``.
            num_layers:
                If given, limit the number of intermediate layers returned.

        Returns:
            List of tensors for each requested layer, each of shape
            ``(batch, time, feature_dim)``, and optional lengths tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms_list = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms_list, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        if self.feature_grad_mult != 1.0:
            x = components.GradMultiply.apply(x, self.feature_grad_mult)
        x_list = self.encoder.extract_features(x, lengths, num_layers)
        return x_list, lengths

    def get_num_params(self) -> int:
        """Calculate current model size (parameters count)."""
        feature_extractor_size, encoder_in_features = (
            self.feature_extractor.get_num_params_and_final_out_channels()
        )
        encoder_size = self.encoder.get_num_params(encoder_in_features)
        return feature_extractor_size + encoder_size

    def forward(  # type: ignore[override]
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """End-to-end forward producing logits from audio.

        Not used directly in wesep separation models; kept for completeness.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms_list = [
                    F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)
                ]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms_list, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])

        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths


def wav2vec2_model_original(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_num_heads: List[int],
    encoder_head_dim: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
    normalize_waveform: bool,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
    use_layerwise_prune: str = False,
) -> Wav2Vec2Model:
    """Build a generic Wav2Vec2-style model."""
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [
            (512, 2, 2)
        ] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode,
        extractor_conv_layer_config,
        extractor_conv_bias,
    )
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        num_heads=encoder_num_heads,
        head_dim=encoder_head_dim,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


def wav2vec2_model(**configs) -> Wav2Vec2Model:
    """Dispatch between generic Wav2Vec2 and WavLM based on config."""
    if "encoder_remaining_heads" in configs:
        return wavlm_model(**configs)
    return wav2vec2_model_original(**configs)


def wavlm_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_use_attention: List[bool],
    encoder_use_feed_forward: List[bool],
    encoder_total_num_heads: List[int],
    encoder_remaining_heads: List[List[int]],
    encoder_num_buckets: int,
    encoder_max_distance: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: List[int],
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int] = None,
    normalize_waveform: bool = False,
    extractor_prune_conv_channels: bool = False,
    encoder_prune_attention_heads: bool = False,
    encoder_prune_attention_layer: bool = False,
    encoder_prune_feed_forward_intermediate: bool = False,
    encoder_prune_feed_forward_layer: bool = False,
    use_layerwise_prune: str = False,
) -> Wav2Vec2Model:
    """Build a WavLM model whose API matches torchaudio's Wav2Vec2Model."""
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [
            (512, 2, 2)
        ] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode,
        extractor_conv_layer_config,
        extractor_conv_bias,
    )
    encoder = components._get_wavlm_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        use_attention=encoder_use_attention,
        use_feed_forward=encoder_use_feed_forward,
        total_num_heads=encoder_total_num_heads,
        remaining_heads=encoder_remaining_heads,
        num_buckets=encoder_num_buckets,
        max_distance=encoder_max_distance,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(normalize_waveform, feature_extractor, encoder, aux)


def wavlm_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    """Convenience constructor for WavLM-Base."""
    return wavlm_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_total_num_heads=[12] * 12,
        encoder_remaining_heads=[list(range(12)) for _ in range(12)],
        encoder_num_buckets=320,
        encoder_max_distance=800,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
        normalize_waveform=False,
    )


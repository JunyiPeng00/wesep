"""Import Hugging Face WavLM / Wav2Vec2 checkpoints into wesep format.

This is adapted from the helper in `wespeaker_hubert` and is intended for
offline conversion of Hugging Face checkpoints into the torchaudio-compatible
format used by :mod:`wesep.wesep.modules.wavlm_frontend.model`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from torch.nn import Module

from ..model import wav2vec2_model, Wav2Vec2Model, wavlm_model

_LG = logging.getLogger(__name__)


def _get_config_wavlm(cfg: Any) -> Dict[str, Any]:
    """Map Hugging Face WavLM config to wavlm_model kwargs."""
    config: Dict[str, Any] = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(
            zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)
        ),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_use_attention": [True] * cfg.num_hidden_layers,
        "encoder_use_feed_forward": [True] * cfg.num_hidden_layers,
        "encoder_total_num_heads": [
            cfg.num_attention_heads for _ in range(cfg.num_hidden_layers)
        ],
        "encoder_remaining_heads": [
            list(range(cfg.num_attention_heads)) for _ in range(cfg.num_hidden_layers)
        ],
        "encoder_num_buckets": cfg.num_buckets,
        "encoder_max_distance": cfg.max_bucket_distance,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": [
            cfg.intermediate_size for _ in range(cfg.num_hidden_layers)
        ],
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
        "normalize_waveform": cfg.feat_extract_norm == "layer",
    }
    return config


def _get_config(cfg: Any) -> Dict[str, Any]:
    """Map Hugging Face Wav2Vec2/HuBERT config to wav2vec2_model kwargs."""
    config: Dict[str, Any] = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(
            zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)
        ),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_use_attention": [True] * cfg.num_hidden_layers,
        "encoder_use_feed_forward": [True] * cfg.num_hidden_layers,
        "encoder_head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "encoder_num_heads": [cfg.num_attention_heads for _ in range(cfg.num_hidden_layers)],
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": [
            cfg.intermediate_size for _ in range(cfg.num_hidden_layers)
        ],
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
        "normalize_waveform": cfg.feat_extract_norm == "layer",
    }
    return config


def _build(config: Dict[str, Any], original: Module) -> Tuple[Wav2Vec2Model, Dict[str, Any]]:
    """Instantiate a Wav2Vec2Model/WavLM and load HF weights."""
    is_for_ctc = original.__class__.__name__ in ["Wav2Vec2ForCTC", "WavLMForCTC"]
    if is_for_ctc:
        aux_num_out = original.config.vocab_size
        wav2vec2 = original.wav2vec2
    else:
        _LG.warning(
            "Model is not an instance of Wav2Vec2ForCTC or WavLMForCTC. "
            '"lm_head" module is not imported.'
        )
        aux_num_out = None
        wav2vec2 = original

    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        imported = wavlm_model(**config, aux_num_out=aux_num_out)
    else:
        imported = wav2vec2_model(**config, aux_num_out=aux_num_out)

    # Feature extractor and projection
    _LG.info("Loading feature extractor and projection.")
    imported.feature_extractor.load_state_dict(
        wav2vec2.feature_extractor.state_dict(), strict=False
    )
    imported.encoder.feature_projection.load_state_dict(
        wav2vec2.feature_projection.state_dict(), strict=False
    )

    # Encoder transformer
    _LG.info("Loading encoder transformer.")
    encoder_state_dict = wav2vec2.encoder.state_dict()
    imported.encoder.transformer.load_state_dict(encoder_state_dict, strict=False)

    if is_for_ctc and imported.aux is not None:
        imported.aux.load_state_dict(original.lm_head.state_dict())

    return imported, config


def import_huggingface_model(original: Module) -> Tuple[Wav2Vec2Model, Dict[str, Any]]:
    """Convert a Hugging Face Wav2Vec2 / WavLM model to wesep format.

    Args:
        original:
            An instance of ``Wav2Vec2ForCTC`` or ``WavLMForCTC`` from
            ``transformers``.

    Returns:
        A tuple of (Wav2Vec2Model, config_dict).
    """
    _LG.info("Importing Hugging Face model.")
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        config = _get_config_wavlm(original.config)
    else:
        config = _get_config(original.config)
    _LG.debug("  - config: %s", config)
    imported, config = _build(config, original)
    return imported, config


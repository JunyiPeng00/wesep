"""Building blocks for WavLM / Wav2Vec2-style SSL models (no pruning)."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple
import math

import torch
from torch import nn, Tensor
from torch.nn import Module


def _init_transformer_params(module: Module) -> None:
    """Initialize Transformer weights as in fairseq."""

    def normal_(data: Tensor) -> None:
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class LayerNorm(nn.LayerNorm):
    """LayerNorm that operates on (B, C, T) with transpose."""

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x


class ConvLayerBlock(Module):
    """Single convolutional block used in the feature extractor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply conv + norm + GELU to 1D waveform features."""
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            length = torch.max(torch.zeros_like(length), length)
        return x, length

    def get_num_params_and_out_channels(self, in_channels: int) -> Tuple[Tensor, int]:
        """Return parameter count and effective output channels."""
        out_channels = self.conv.out_channels

        num_params = in_channels * out_channels * self.kernel_size
        if self.conv.bias is not None:
            num_params += out_channels
        if self.layer_norm is not None:
            num_params += out_channels * 2

        return torch.tensor(num_params), out_channels


class FeatureExtractor(Module):
    """CNN feature extractor from raw waveform."""

    def __init__(self, conv_layers: nn.ModuleList) -> None:
        super().__init__()
        self.conv_layers = conv_layers
        self.dummy_weight = nn.Parameter(
            torch.ones(conv_layers[-1].conv.out_channels, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Map (B, T) waveform to (B, T', C) features."""
        if x.ndim != 2:
            raise ValueError(
                f"Expected input tensor to be 2D (batch, time), but received {list(x.shape)}"
            )

        x = x.unsqueeze(1)  # (B, 1, T)
        for layer in self.conv_layers:
            x, length = layer(x, length)
        x = x.transpose(1, 2)  # (B, T', C)
        x = x * self.dummy_weight
        return x, length

    def get_num_params_and_final_out_channels(self) -> Tuple[int, int]:
        """Return parameter count and final channel dim."""
        in_channels = 1
        num_params = 0
        for layer in self.conv_layers:
            layer_params, in_channels = layer.get_num_params_and_out_channels(in_channels)
            num_params += int(layer_params)
        num_params += in_channels  # dummy weight
        return num_params, in_channels

class FeatureProjection(Module):
    """Projection layer between CNN features and Transformer encoder."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

    def get_num_params(self, in_features: int) -> int:
        return in_features * 2 + (in_features + 1) * self.projection.out_features


class ConvolutionalPositionalEmbedding(Module):
    """Conv positional embedding layer at the Transformer input."""

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self) -> "ConvolutionalPositionalEmbedding":
        for hook in self.conv._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    """Multi-head self-attention used in Wav2Vec2-style encoders."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = torch.nn.Dropout(dropout)

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"Expected input shape (batch, sequence, embed_dim={self.embed_dim}), "
                f"found {x.shape}."
            )
        batch_size, length, _ = x.size()

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(*shape).transpose(2, 1)

        weights = (self.scaling * q) @ k
        if attention_mask is not None:
            weights += attention_mask
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        output = output.transpose(2, 1).reshape(
            batch_size, length, self.num_heads * self.head_dim
        )
        output = self.out_proj(output)

        return output, None

    def get_num_params(self) -> int:
        """Return an approximate parameter count."""
        num_heads = self.num_heads
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * 3 + (
            num_heads * self.head_dim + 1
        ) * self.embed_dim
        return num_params


class WavLMSelfAttention(SelfAttention):
    """Self-attention with relative position bias for WavLM."""

    def __init__(
        self,
        embed_dim: int,
        total_num_heads: int,
        remaining_heads: Optional[List[int]] = None,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = True,
    ) -> None:
        self.total_num_heads = total_num_heads
        if remaining_heads is None:
            self.remaining_heads = list(range(total_num_heads))
        else:
            self.remaining_heads = remaining_heads

        self.head_dim = embed_dim // total_num_heads

        super().__init__(
            embed_dim=embed_dim,
            num_heads=len(self.remaining_heads),
            head_dim=self.head_dim,
            dropout=dropout,
        )

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, total_num_heads)
        else:
            self.rel_attn_embed = None

        self.k_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(len(self.remaining_heads) * self.head_dim, embed_dim, bias=bias)

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, total_num_heads, 1, 1))
        self.has_position_bias = True

    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        """Compute relative position embeddings for WavLM."""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(
        self, relative_positions: Tensor, bidirectional: bool = True
    ) -> Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(
                relative_positions, torch.zeros_like(relative_positions)
            )

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key_padding_mask is None

        if self.rel_attn_embed is not None and position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(
                bsz * self.total_num_heads, seq_len, seq_len
            )

        attn_mask_rel_pos: Optional[Tensor] = None
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:
                query_layer = query.view(bsz, seq_len, self.total_num_heads, -1)
                query_layer = query_layer.permute(0, 2, 1, 3)

                gate_a, gate_b = torch.sigmoid(
                    self.gru_rel_pos_linear(query_layer)
                    .view(bsz, self.total_num_heads, seq_len, 2, 4)
                    .sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.total_num_heads, -1, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view((-1, seq_len, seq_len))
            attn_mask_rel_pos = attn_mask_rel_pos.reshape(
                bsz, self.total_num_heads, seq_len, seq_len
            )[:, self.remaining_heads, :, :]

        attn_mask = attn_mask_rel_pos
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.reshape(bsz, 1, 1, seq_len), float("-inf")
            )
        attn_output, _ = super().forward(query, attention_mask=attn_mask)

        return attn_output, position_bias


class FeedForward(Module):
    """Position-wise FFN following attention in encoder layers."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)

        return x

    def get_num_params(self) -> int:
        io_features = self.intermediate_dense.in_features
        intermediate_features = self.intermediate_dense.out_features
        num_params = (io_features + 1) * intermediate_features + (
            intermediate_features + 1
        ) * io_features
        return num_params


class EncoderLayer(Module):
    """Single transformer encoder layer combining attention and FFN."""

    def __init__(
        self,
        attention: Optional[Module],
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Optional[Module],
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.attention is not None:
            residual = x

            if self.layer_norm_first:
                x = self.layer_norm(x)

            x, position_bias = self.attention(
                x,
                attention_mask=attention_mask,
                position_bias=position_bias,
                key_padding_mask=key_padding_mask,
            )

            x = self.dropout(x)
            x = residual + x

        if self.layer_norm_first:
            if self.feed_forward is not None:
                x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            if self.feed_forward is not None:
                x = x + self.feed_forward(x)
            x = self.final_layer_norm(x)
        return x, position_bias

    def get_num_params(self) -> int:
        num_params = self.embed_dim * 2 * 2
        if self.attention is not None:
            num_params += self.attention.get_num_params()
        if self.feed_forward is not None:
            num_params += self.feed_forward.get_num_params()
        return num_params


class Transformer(Module):
    """Stack of encoder layers with positional convolution."""

    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ) -> None:
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor) -> Tensor:
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
        position_bias: Optional[Tensor] = None,
    ) -> List[Tensor]:
        if num_layers is not None and not 0 < num_layers <= len(self.layers):
            raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        x = self._preprocess(x)
        ret.append(x)

        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret

    def get_num_params(self) -> int:
        num_params = sum(p.numel() for p in self.pos_conv_embed.parameters()) + (
            self.pos_conv_embed.embed_dim * 2
        )
        for layer in self.layers:
            num_params += layer.get_num_params()
        return num_params


class Encoder(Module):
    """Full encoder: projection + Transformer stack."""

    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ) -> None:
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[
                :, None
            ]
            x[mask] = 0.0
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(  # type: ignore[override]
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        interm = self.transformer.get_intermediate_outputs(
            x, attention_mask=masks, num_layers=num_layers
        )
        return interm

    def get_num_params(self, in_features: int) -> int:
        feature_projection_size = self.feature_projection.get_num_params(in_features)
        transformer_size = self.transformer.get_num_params()
        return feature_projection_size + transformer_size


################################################################################
def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
) -> FeatureExtractor:
    """Construct the CNN feature extractor (no pruning)."""
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization: Optional[Module] = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
            )
        )
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks))


def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    num_heads: List[int],
    head_dim: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
) -> Encoder:
    """Construct standard Wav2Vec2 encoder.

    All pruning flags are wired but not used in wesep.
    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    encoder_layers = nn.ModuleList()
    for idx in range(num_layers):
        if use_attention[idx]:
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads[idx],
                head_dim=head_dim,
                dropout=attention_dropout,
            )
        else:
            attention = None
        if use_feed_forward[idx]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[idx],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _get_wavlm_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    total_num_heads: List[int],
    remaining_heads: List[List[int]],
    num_buckets: int,
    max_distance: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
) -> Encoder:
    """Construct WavLM encoder with relative positional attention."""
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        if use_attention[i]:
            attention = WavLMSelfAttention(
                embed_dim=embed_dim,
                total_num_heads=total_num_heads[i],
                remaining_heads=remaining_heads[i],
                dropout=attention_dropout,
                has_relative_attention_bias=(i == 0),
                num_buckets=num_buckets,
                max_distance=max_distance,
            )
        else:
            attention = None
        if use_feed_forward[i]:
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[i],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
            )
        else:
            feed_forward = None
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate padding mask given padded input and lengths."""
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[
        :, None
    ]
    return mask


class GradMultiply(torch.autograd.Function):
    """Gradient scaling helper used by feature_grad_mult."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:  # type: ignore[override]
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        return grad * ctx.scale, None


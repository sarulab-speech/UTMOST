"""WavLM encoder for inference, numerically matching Hugging Face Transformers.

Adapted from transformers/models/wavlm/modeling_wavlm.py (Apache License 2.0,
Copyright 2021 The Fairseq Authors, Microsoft Research, and The HuggingFace Inc. team).
See THIRD_PARTY_NOTICES.md. Only the configuration used by UTMOST (WavLM-Large:
layer-normalized convolutional feature encoder, pre-LN transformer with gated relative
position bias) is kept. Dropout, SpecAugment masking, and training utilities are removed.
Parameter names follow the original implementation, so its state dicts load unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import WeightNormConv1d


class ConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int, bias: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, bias=bias)
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.layer_norm(x.transpose(-2, -1)).transpose(-2, -1)
        return F.gelu(x)


class FeatureEncoder(nn.Module):
    def __init__(self, conv_dim: list[int], conv_kernel: list[int], conv_stride: list[int], conv_bias: bool) -> None:
        super().__init__()
        dims = [1, *conv_dim]
        self.conv_layers = nn.ModuleList(
            ConvLayer(dims[i], dims[i + 1], conv_kernel[i], conv_stride[i], conv_bias) for i in range(len(conv_dim))
        )

    def forward(self, input_values: Tensor) -> Tensor:
        x = input_values[:, None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, eps: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=eps)
        self.projection = nn.Linear(in_dim, hidden)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.layer_norm(x))


class PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.conv = WeightNormConv1d(hidden, hidden, kernel, padding=kernel // 2, groups=groups, dim=2)
        self.num_pad_remove = 1 if kernel % 2 == 0 else 0

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x.transpose(1, 2))
        if self.num_pad_remove:
            x = x[:, :, : -self.num_pad_remove]
        return F.gelu(x).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention with WavLM's gated relative position bias."""

    def __init__(self, dim: int, heads: int, num_buckets: int, max_distance: int, has_relative_position_bias: bool) -> None:
        super().__init__()
        self.embed_dim, self.num_heads = dim, heads
        self.head_dim = dim // heads
        self.num_buckets, self.max_distance = num_buckets, max_distance
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
        if has_relative_position_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, heads)

    def _relative_positions_bucket(self, relative_positions: Tensor) -> Tensor:
        num_buckets = self.num_buckets // 2
        relative_buckets = (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        if_large = torch.log(relative_positions.float() / max_exact)
        if_large = if_large / math.log(self.max_distance / max_exact)
        if_large = if_large * (num_buckets - max_exact)
        if_large = (max_exact + if_large).to(torch.long)
        if_large = torch.min(if_large, torch.full_like(if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_positions, if_large)

    def compute_bias(self, length: int) -> Tensor:
        context = torch.arange(length, dtype=torch.long)[:, None]
        memory = torch.arange(length, dtype=torch.long)[None, :]
        buckets = self._relative_positions_bucket(memory - context).to(self.rel_attn_embed.weight.device)
        return self.rel_attn_embed(buckets).permute([2, 0, 1])

    def forward(self, x: Tensor, key_padding_mask: Tensor | None, position_bias: Tensor | None) -> tuple[Tensor, Tensor]:
        bsz, length, _ = x.size()
        if position_bias is None:
            position_bias = self.compute_bias(length)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, length, length)

        gated = x.view(x.shape[:-1] + (self.num_heads, -1)).permute(0, 2, 1, 3)
        proj = self.gru_rel_pos_linear(gated).view(gated.shape[:-1] + (2, 4)).sum(-1)
        gate_a, gate_b = torch.sigmoid(proj).chunk(2, dim=-1)
        gate = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
        gated_bias = (gate.view(bsz * self.num_heads, -1, 1) * position_bias).view((-1, length, length))

        q = x.transpose(0, 1)
        out, _ = F.multi_head_attention_forward(
            q,
            q,
            q,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None,
            None,
            False,
            0.0,
            self.out_proj.weight,
            self.out_proj.bias,
            training=False,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=gated_bias,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
        return out.transpose(0, 1), position_bias


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, inner)
        self.output_dense = nn.Linear(inner, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class EncoderLayer(nn.Module):
    """Pre-LN transformer layer (`do_stable_layer_norm=True`)."""

    def __init__(self, cfg: dict[str, Any], has_relative_position_bias: bool) -> None:
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["layer_norm_eps"]
        self.attention = Attention(
            dim, cfg["num_attention_heads"], cfg["num_buckets"], cfg["max_bucket_distance"], has_relative_position_bias
        )
        self.layer_norm = nn.LayerNorm(dim, eps=eps)
        self.feed_forward = FeedForward(dim, cfg["intermediate_size"])
        self.final_layer_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None, position_bias: Tensor | None) -> tuple[Tensor, Tensor]:
        attn, position_bias = self.attention(self.layer_norm(x), key_padding_mask, position_bias)
        x = x + attn
        return x + self.feed_forward(self.final_layer_norm(x)), position_bias


class Encoder(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        dim = cfg["hidden_size"]
        self.pos_conv_embed = PositionalConvEmbedding(dim, cfg["num_conv_pos_embeddings"], cfg["num_conv_pos_embedding_groups"])
        self.layer_norm = nn.LayerNorm(dim, eps=cfg["layer_norm_eps"])
        self.layers = nn.ModuleList(EncoderLayer(cfg, i == 0) for i in range(cfg["num_hidden_layers"]))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        key_padding_mask = None
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            key_padding_mask = torch.zeros(mask.shape, dtype=x.dtype, device=x.device).masked_fill(~mask, float("-inf"))
        x = x + self.pos_conv_embed(x)
        position_bias = None
        for layer in self.layers:
            x, position_bias = layer(x, key_padding_mask, position_bias)
        return self.layer_norm(x)


class WavLM(nn.Module):
    """WavLM encoder returning the last hidden states, as `WavLMModel(...).last_hidden_state`."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        unsupported = {
            "feat_extract_norm": ("layer", cfg.get("feat_extract_norm")),
            "do_stable_layer_norm": (True, cfg.get("do_stable_layer_norm")),
            "feat_extract_activation": ("gelu", cfg.get("feat_extract_activation", "gelu")),
            "hidden_act": ("gelu", cfg.get("hidden_act", "gelu")),
            "add_adapter": (False, cfg.get("add_adapter", False)),
        }
        for key, (expected, got) in unsupported.items():
            if got != expected:
                raise ValueError(f"WavLM config {key}={got!r} is not supported (expected {expected!r})")
        self.conv_kernel, self.conv_stride = list(cfg["conv_kernel"]), list(cfg["conv_stride"])
        self.feature_extractor = FeatureEncoder(cfg["conv_dim"], self.conv_kernel, self.conv_stride, cfg.get("conv_bias", False))
        self.feature_projection = FeatureProjection(cfg["conv_dim"][-1], cfg["hidden_size"], cfg["layer_norm_eps"])
        # Mask embedding of the pre-trained model; kept so that its state dict loads, never used at inference.
        self.masked_spec_embed = nn.Parameter(torch.zeros(cfg["hidden_size"]))
        self.encoder = Encoder(cfg)

    def output_lengths(self, input_lengths: Tensor) -> Tensor:
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            input_lengths = torch.div(input_lengths - kernel, stride, rounding_mode="floor") + 1
        return input_lengths

    def forward(self, input_values: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        features = self.feature_extractor(input_values).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            lengths = self.output_lengths(attention_mask.sum(dim=-1)).to(torch.long)
            mask = torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None]
        return self.encoder(self.feature_projection(features), mask)

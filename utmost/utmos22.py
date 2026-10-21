"""UTMOS22 strong learner, used by UTMOST as a frozen side feature.

Adapted from SpeechMOS v1.2.0 (https://github.com/tarepan/SpeechMOS, MIT License,
Copyright (c) 2023 tarepan), whose wav2vec 2.0 code is derived from fairseq (MIT License,
Copyright Facebook, Inc. and its affiliates). The weights are the official UTMOS22 strong
learner (https://github.com/sarulab-speech/UTMOS22, MIT License, Copyright 2022
Saruwatari & Koyama Laboratory, The University of Tokyo). See THIRD_PARTY_NOTICES.md.

Changes from SpeechMOS: `embed` returns the frame-averaged BLSTM output next to the score,
the positional convolution uses `WeightNormConv1d`, and dropout layers are removed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import WeightNormConv1d


class UTMOS22Strong(nn.Module):
    """UTMOS strong learner without the phoneme encoder (16 kHz input)."""

    embedding_dim = 1025

    def __init__(self) -> None:
        super().__init__()
        feat_ssl, feat_domain, feat_judge, feat_rnn, feat_proj = 768, 128, 128, 512, 2048
        self.wav2vec2 = Wav2Vec2Model()
        self.domain_emb = nn.Parameter(torch.empty(1, feat_domain), requires_grad=False)
        self.judge_emb = nn.Parameter(torch.empty(1, feat_judge), requires_grad=False)
        self.blstm = nn.LSTM(feat_ssl + feat_domain + feat_judge, feat_rnn, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(feat_rnn * 2, feat_proj), nn.ReLU(), nn.Linear(feat_proj, 1))

    def embed(self, wave: Tensor) -> Tensor:
        """(1, T) 16 kHz waveform -> (1, 1025): mean BLSTM output [1024] and UTMOS score [1]."""
        units = self.wav2vec2(wave)
        batch, frames, _ = units.size()
        domain = self.domain_emb.unsqueeze(1).expand(batch, frames, -1).to(units.dtype)
        judge = self.judge_emb.unsqueeze(1).expand(batch, frames, -1).to(units.dtype)
        hidden = self.blstm(torch.cat([units, domain, judge], dim=2))[0]
        score = self.projection(hidden).mean(dim=1) * 2.0 + 3.0
        return torch.cat([hidden.mean(dim=1), score], dim=1)

    def forward(self, wave: Tensor) -> Tensor:
        """(1, T) 16 kHz waveform -> (1,) UTMOS22 score."""
        return self.embed(wave)[:, -1]


class Wav2Vec2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        feat_h1, feat_h2 = 512, 768
        layers = [(feat_h1, 10, 5)] + [(feat_h1, 3, 2)] * 4 + [(feat_h1, 2, 2)] * 2
        self.feature_extractor = ConvFeatureExtractionModel(layers)
        self.layer_norm = nn.LayerNorm(feat_h1)
        self.post_extract_proj = nn.Linear(feat_h1, feat_h2)
        self.encoder = TransformerEncoder(feat_h2)
        # Mask embedding of the pre-trained model; kept so that its state dict loads, never used at inference.
        self.mask_emb = nn.Parameter(torch.zeros(feat_h2))

    def forward(self, source: Tensor) -> Tensor:
        features = self.feature_extractor(source).transpose(1, 2)
        return self.encoder(self.post_extract_proj(self.layer_norm(features)))


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers: list[tuple[int, int, int]]) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_d = 1
        for i, (dim, k, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_d, dim, k, stride=stride, bias=False)
            if i == 0:
                # Indices match the original nn.Sequential(conv, dropout, norm, activation).
                self.conv_layers.append(nn.Sequential(conv, nn.Identity(), nn.GroupNorm(dim, dim, affine=True), nn.GELU()))
            else:
                self.conv_layers.append(nn.Sequential(conv, nn.Identity(), nn.GELU()))
            in_d = dim

    def forward(self, series: Tensor) -> Tensor:
        series = series.unsqueeze(1)
        for conv in self.conv_layers:
            series = conv(series)
        return series


class TransformerEncoder(nn.Module):
    def __init__(self, feat: int) -> None:
        super().__init__()
        self.required_seq_len_multiple = 2
        self.pos_conv = nn.Sequential(
            WeightNormConv1d(feat, feat, kernel_size=128, padding=64, groups=16, dim=2),
            SamePad(),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(feat)
        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(feat, 3072, 12) for _ in range(12)])

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x)
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple)
        key_padding_mask = None
        if pad_length > 0:
            key_padding_mask = x.new_zeros((x.size(0), x.size(1)))
            key_padding_mask[:, -pad_length:] = float("-inf")
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = x.transpose(0, 1)
        if pad_length > 0:
            x = x[:, :-pad_length]
        return x


class SamePad(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-1]


def pad_to_multiple(x: Tensor, multiple: int) -> tuple[Tensor, int]:
    """Pad the time axis (dim -2) at the tail to a multiple of `multiple`."""
    length = x.size(-2)
    remainder = math.ceil(length / multiple) * multiple - length
    if remainder == 0:
        return x, 0
    return F.pad(x, (0, 0, 0, remainder), value=0.0), remainder


class TransformerSentenceEncoderLayer(nn.Module):
    """Post-LN transformer layer (fairseq `layer_norm_first=False`)."""

    def __init__(self, feat: int, ffn: int, heads: int) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(feat, heads)
        self.fc1 = nn.Linear(feat, ffn)
        self.fc2 = nn.Linear(ffn, feat)
        self.self_attn_layer_norm = nn.LayerNorm(feat)
        self.final_layer_norm = nn.LayerNorm(feat)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, key_padding_mask))
        return self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        """x: (T, B, D); key_padding_mask: (B, T), -inf at padded positions."""
        return F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=False,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )[0]

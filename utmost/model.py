"""UTMOST network: WavLM-Large encoder, frozen UTMOS22 side feature, and listener-conditioned heads.

Inference path of the training code for the released configuration. Parameter names match
the training checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utmos22 import UTMOS22Strong
from .wavlm import WavLM

EMOTIONS = ("Angry", "Happy", "Neutral", "Sad", "Surprise")


@dataclass
class UTMOSTConfig:
    wavlm: dict[str, Any] = field(default_factory=dict)
    num_listeners: int = 262
    listener_embedding_dim: int = 64
    target_embedding_dim: int = 32
    frame_hidden_dim: int = 512
    utmos_feature_dim: int = 1025
    utmos_feature_proj_dim: int = 64
    cat_embedding_dim: int = 64
    sphere_classes: int = 9
    dropout: float = 0.15
    sample_rate: int = 16000
    max_duration_sec: float = 8.0
    listener_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UTMOSTConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(values.dtype).unsqueeze(-1)
    return (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def masked_stats_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    mean = masked_mean(hidden, mask)
    mask_f = mask.to(hidden.dtype).unsqueeze(-1)
    var = ((hidden - mean.unsqueeze(1)).pow(2) * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    return torch.cat([mean, torch.sqrt(var.clamp_min(1e-6))], dim=-1)


def bound_1_to_5(raw: Tensor) -> Tensor:
    return 1.0 + 4.0 * torch.sigmoid(raw)


class UTMOSTNetwork(nn.Module):
    def __init__(self, config: UTMOSTConfig) -> None:
        super().__init__()
        self.config = config
        hid, lis, tgt, cat = (
            config.frame_hidden_dim,
            config.listener_embedding_dim,
            config.target_embedding_dim,
            config.cat_embedding_dim,
        )
        self.backbone = WavLM(config.wavlm)
        ssl_hidden = config.wavlm["hidden_size"]
        self.utmos22 = UTMOS22Strong()
        self.utmos_proj = nn.Sequential(nn.Linear(config.utmos_feature_dim, config.utmos_feature_proj_dim), nn.ReLU())
        self.listener_embedding = nn.Embedding(config.num_listeners, lis)
        self.target_embedding = nn.Embedding(len(EMOTIONS), tgt)
        self.acoustic_encoder = nn.LSTM(
            ssl_hidden + config.utmos_feature_proj_dim, hid // 2, batch_first=True, bidirectional=True
        )
        self.qmos_frame_head = make_mlp(hid + lis, hid, 1, config.dropout)
        cat_pooled_dim = (hid + lis) * 2
        self.cat_projector = nn.Sequential(
            nn.LayerNorm(cat_pooled_dim),
            nn.Dropout(config.dropout),
            nn.Linear(cat_pooled_dim, cat),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.cat_classifier = nn.Linear(cat, len(EMOTIONS))
        self.emotion_encoder = nn.LSTM(hid + lis + tgt + cat, hid // 2, batch_first=True, bidirectional=True)
        self.vad_head = make_mlp(hid * 2, hid, 3, config.dropout)
        self.emos_frame_head = make_mlp(hid, hid, 1, config.dropout)
        # Auxiliary training head (valence-arousal-dominance region); not used at inference.
        self.sphere_head = make_mlp(hid * 2, hid, config.sphere_classes, config.dropout)

    def _run_lstm(self, module: nn.LSTM, hidden: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(module(packed)[0], batch_first=True, total_length=hidden.size(1))
        return out

    def encode(self, input_values: Tensor, attention_mask: Tensor, utmos_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Normalized, padded 16 kHz waveforms -> acoustic frames (B, F, 512) and frame mask (B, F)."""
        ssl = self.backbone(input_values, attention_mask)
        lengths = self.backbone.output_lengths(attention_mask.sum(dim=1))
        mask = torch.arange(ssl.size(1), device=ssl.device).unsqueeze(0) < lengths.unsqueeze(1)
        side = self.utmos_proj(utmos_feature.to(ssl.dtype)).unsqueeze(1).expand(-1, ssl.size(1), -1)
        acoustic = self._run_lstm(self.acoustic_encoder, torch.cat([ssl, side], dim=-1), mask)
        return acoustic, mask

    def _joint(self, acoustic: Tensor, listener_emb: Tensor) -> Tensor:
        return torch.cat([acoustic, listener_emb.unsqueeze(1).expand(-1, acoustic.size(1), -1)], dim=-1)

    def quality_and_category(self, acoustic: Tensor, mask: Tensor, listener_emb: Tensor) -> tuple[Tensor, Tensor]:
        """One listener per row: qMOS (1-5) and category probabilities. Neither reads the requested emotion."""
        joint = self._joint(acoustic, listener_emb)
        qmos_frame = bound_1_to_5(self.qmos_frame_head(joint).squeeze(-1))
        qmos = masked_mean(qmos_frame.unsqueeze(-1), mask).squeeze(-1)
        cat_probs = torch.sigmoid(self.cat_classifier(self.cat_projector(masked_stats_pool(joint, mask))))
        return qmos, cat_probs

    def emotion_branch(self, acoustic: Tensor, mask: Tensor, target_emotion_id: Tensor, listener_emb: Tensor) -> tuple[Tensor, Tensor]:
        """One listener and one requested emotion per row: eMOS (1-5) and valence, arousal, dominance (1-5)."""
        frames = acoustic.size(1)
        joint = self._joint(acoustic, listener_emb)
        cat_embedding = self.cat_projector(masked_stats_pool(joint, mask))
        target_frame = self.target_embedding(target_emotion_id).unsqueeze(1).expand(-1, frames, -1)
        cat_frame = cat_embedding.unsqueeze(1).expand(-1, frames, -1)
        emotion_hidden = self._run_lstm(self.emotion_encoder, torch.cat([joint, target_frame, cat_frame], dim=-1), mask)
        vad = bound_1_to_5(self.vad_head(masked_stats_pool(emotion_hidden, mask)))
        emos_frame = bound_1_to_5(self.emos_frame_head(emotion_hidden).squeeze(-1))
        emos = masked_mean(emos_frame.unsqueeze(-1), mask).squeeze(-1)
        return emos, vad

    @staticmethod
    def _per_listener(x: Tensor, n: int) -> Tensor:
        """(rows, ...) -> (rows * n, ...), each row repeated for n listeners."""
        return x.unsqueeze(1).expand(x.size(0), n, *x.shape[1:]).reshape(x.size(0) * n, *x.shape[1:])

    def _listener_embeddings(self, ids: Tensor, rows: int) -> Tensor:
        return self.listener_embedding(ids).unsqueeze(0).expand(rows, ids.numel(), -1).reshape(rows * ids.numel(), -1)

    def average_quality_and_category(
        self, acoustic: Tensor, mask: Tensor, listener_ids: Tensor, chunk_size: int = 16
    ) -> tuple[Tensor, Tensor]:
        """qMOS (B,) and category probabilities (B, 5), averaged over listener embeddings (Eq. 1 of the paper)."""
        rows = acoustic.size(0)
        qmos = cat = None
        for ids in listener_ids.split(chunk_size):
            n = ids.numel()
            q, c = self.quality_and_category(
                self._per_listener(acoustic, n), self._per_listener(mask, n), self._listener_embeddings(ids, rows)
            )
            q, c = q.reshape(rows, n).float().sum(dim=1), c.reshape(rows, n, -1).float().sum(dim=1)
            qmos, cat = (q, c) if qmos is None else (qmos + q, cat + c)
        return qmos / listener_ids.numel(), cat / listener_ids.numel()

    def average_emotion_branch(
        self, acoustic: Tensor, mask: Tensor, target_emotion_ids: Tensor, listener_ids: Tensor, chunk_size: int = 16
    ) -> tuple[Tensor, Tensor]:
        """eMOS (B, K) and VAD (B, K, 3) for K requested emotions per utterance (`target_emotion_ids` is (B, K)),
        averaged over listener embeddings. All B * K conditions run in one pass per listener chunk."""
        batch, k = target_emotion_ids.shape
        acoustic = self._per_listener(acoustic, k)
        mask = self._per_listener(mask, k)
        targets = target_emotion_ids.reshape(-1)
        rows = batch * k
        emos = vad = None
        for ids in listener_ids.split(chunk_size):
            n = ids.numel()
            e, v = self.emotion_branch(
                self._per_listener(acoustic, n),
                self._per_listener(mask, n),
                self._per_listener(targets, n),
                self._listener_embeddings(ids, rows),
            )
            e, v = e.reshape(rows, n).float().sum(dim=1), v.reshape(rows, n, -1).float().sum(dim=1)
            emos, vad = (e, v) if emos is None else (emos + e, vad + v)
        total = listener_ids.numel()
        return (emos / total).reshape(batch, k), (vad / total).reshape(batch, k, -1)

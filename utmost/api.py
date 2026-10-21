"""High-level UTMOST interface: load once, score files or waveforms."""

from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .audio import AudioInput, load_audio
from .layers import rename_weight_norm_keys
from .model import EMOTIONS, UTMOSTConfig, UTMOSTNetwork

DEFAULT_REPO = "sarulab-speech/UTMOST"
CUDA_TORCH_INSTALL = "pip install torch --index-url https://download.pytorch.org/whl/cu130"
WEIGHT_FILES = {"fp32": "model.safetensors", "fp16": "model.fp16.safetensors"}
EMOTION_ALIASES = {
    "angry": "Angry",
    "anger": "Angry",
    "happy": "Happy",
    "happiness": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
}


def normalize_emotion(value: str) -> str:
    key = value.strip().lower()
    if key not in EMOTION_ALIASES:
        raise ValueError(f"unknown emotion {value!r}; expected one of {', '.join(EMOTIONS)}")
    return EMOTION_ALIASES[key]


def _resolve_files(name_or_path: str | os.PathLike, precision: str, revision: str | None) -> tuple[Path, Path]:
    local = Path(name_or_path)
    if local.is_dir():
        weights = local / WEIGHT_FILES[precision]
        if not weights.exists() and precision == "fp16":
            weights = local / WEIGHT_FILES["fp32"]
        return local / "config.json", weights
    from huggingface_hub import hf_hub_download

    try:
        config = hf_hub_download(str(name_or_path), "config.json", revision=revision)
        weights = hf_hub_download(str(name_or_path), WEIGHT_FILES[precision], revision=revision)
    except Exception as err:
        status = getattr(getattr(err, "response", None), "status_code", None)
        if type(err).__name__ in ("RepositoryNotFoundError", "GatedRepoError") or status in (401, 403):
            login = (
                "log in with an account that can read it: run `hf auth login`, "
                "run `from huggingface_hub import login; login()` in Python, or set HF_TOKEN."
            )
            if str(name_or_path) == DEFAULT_REPO:
                raise RuntimeError(
                    "The UTMOST weights are not public yet. They were trained on the VoiceMOS Challenge 2026 "
                    "Track 2 dataset, whose rights are held by the challenge organizers, and will be released at "
                    f"https://huggingface.co/{DEFAULT_REPO} once the organizers make the dataset public. "
                    f"If you have been given access, {login}"
                ) from err
            raise RuntimeError(
                f"Cannot access the Hugging Face repository {str(name_or_path)!r} (HTTP {status}). "
                f"If it is private, {login}"
            ) from err
        raise
    return Path(config), Path(weights)


def _warn_if_gpu_unused() -> None:
    """Warn when an NVIDIA GPU is present but the installed PyTorch has no CUDA support."""
    if torch.version.cuda is not None or shutil.which("nvidia-smi") is None:
        return
    where = "On Windows, pip installs the CPU-only PyTorch by default. " if sys.platform == "win32" else ""
    warnings.warn(
        "An NVIDIA GPU was found, but the installed PyTorch has no CUDA support, so UTMOST runs on the CPU. "
        f"{where}To use the GPU, replace it with a CUDA build: `pip uninstall -y torch` and then "
        f"`{CUDA_TORCH_INSTALL}` (other CUDA versions: https://pytorch.org/get-started/locally/).",
        stacklevel=3,
    )


class UTMOST(nn.Module):
    """Multi-task model for assessing emotional speech synthesis.

    One forward pass returns quality MOS (qMOS), emotion-similarity MOS (eMOS) for a
    requested emotion, valence, arousal, dominance, and the probability that a listener
    selects each of the five perceived emotion categories. Predictions are averaged over
    64 listener embeddings.
    """

    def __init__(self, config: UTMOSTConfig) -> None:
        super().__init__()
        self.config = config
        self.network = UTMOSTNetwork(config)
        self.register_buffer("listener_ids", torch.tensor(config.listener_ids, dtype=torch.long), persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike = DEFAULT_REPO,
        device: str | torch.device | None = None,
        fp16: bool = True,
        revision: str | None = None,
    ) -> "UTMOST":
        """Load UTMOST from a Hugging Face Hub repository or a local directory.

        By default the half-precision weights are used, and inference runs in float16 on
        CUDA and in float32 on other devices. With `fp16=False` the single-precision
        weights are used and inference runs in float32.
        """
        from safetensors.torch import load_file

        if device is None:
            if not torch.cuda.is_available():
                _warn_if_gpu_unused()
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        config_path, weights_path = _resolve_files(name_or_path, "fp16" if fp16 else "fp32", revision)
        config = UTMOSTConfig.from_dict(json.loads(Path(config_path).read_text()))
        model = cls(config)
        state = rename_weight_norm_keys(load_file(str(weights_path)))
        model.network.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=True)
        compute = torch.float16 if fp16 and device.type == "cuda" else torch.float32
        return model.to(device=device, dtype=compute).eval()

    @property
    def device(self) -> torch.device:
        return self.listener_ids.device

    @property
    def dtype(self) -> torch.dtype:
        return self.network.listener_embedding.weight.dtype

    def _prepare(self, wave: Tensor) -> tuple[Tensor, Tensor]:
        """Full waveform for UTMOS22; centre `max_duration_sec`, zero-mean unit-variance waveform for WavLM."""
        max_samples = int(self.config.sample_rate * self.config.max_duration_sec)
        cropped = wave
        if cropped.numel() > max_samples:
            start = (cropped.numel() - max_samples) // 2
            cropped = cropped[start : start + max_samples]
        cropped = (cropped - cropped.mean()) / torch.sqrt(cropped.var(unbiased=False) + 1e-7)
        return wave, cropped

    @torch.inference_mode()
    def _score_batch(
        self, waves: Sequence[Tensor], emotions: Sequence[str | None], chunk_size: int, all_emotions: bool
    ) -> list[dict[str, Any]]:
        net, device, dtype = self.network, self.device, self.dtype
        side, inputs = [], []
        for wave in waves:
            full, cropped = self._prepare(wave)
            side.append(net.utmos22.embed(full.to(device=device, dtype=dtype).unsqueeze(0)))
            inputs.append(cropped)
        lengths = torch.tensor([x.numel() for x in inputs])
        input_values = pad_sequence(inputs, batch_first=True).to(device=device, dtype=dtype)
        attention_mask = (torch.arange(input_values.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).long().to(device)
        acoustic, mask = net.encode(input_values, attention_mask, torch.cat(side, dim=0))

        # qMOS and the category probabilities do not read the requested emotion. Inputs without a
        # requested emotion use the category with the highest predicted probability.
        qmos, cat_probs = net.average_quality_and_category(acoustic, mask, self.listener_ids, chunk_size)
        given = [EMOTIONS.index(e) if e is not None else None for e in emotions]
        predicted = cat_probs.argmax(dim=-1).tolist()
        resolved = [g if g is not None else p for g, p in zip(given, predicted)]
        if all_emotions:
            targets = torch.arange(len(EMOTIONS), device=device).repeat(len(waves), 1)
            columns = resolved
        else:
            targets = torch.tensor(resolved, device=device).unsqueeze(1)
            columns = [0] * len(waves)
        emos, vad = net.average_emotion_branch(acoustic, mask, targets, self.listener_ids, chunk_size)

        results = []
        for row, (emo, col) in enumerate(zip(resolved, columns)):
            out = {
                "qmos": float(qmos[row]),
                "emos": float(emos[row, col]),
                "emotion": EMOTIONS[emo],
                "emotion_source": "given" if given[row] is not None else "predicted",
                "valence": float(vad[row, col, 0]),
                "arousal": float(vad[row, col, 1]),
                "dominance": float(vad[row, col, 2]),
                "category": {name: float(p) for name, p in zip(EMOTIONS, cat_probs[row])},
            }
            if all_emotions:
                out["by_emotion"] = {
                    name: {
                        "emos": float(emos[row, k]),
                        "valence": float(vad[row, k, 0]),
                        "arousal": float(vad[row, k, 1]),
                        "dominance": float(vad[row, k, 2]),
                    }
                    for k, name in enumerate(EMOTIONS)
                }
            results.append(out)
        return results

    def predict(
        self,
        audio: AudioInput | Sequence[AudioInput],
        emotion: str | Sequence[str | None] | None = None,
        sample_rate: int | None = None,
        batch_size: int = 8,
        listener_chunk_size: int = 16,
        all_emotions: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Score one input or a list of inputs.

        Args:
            audio: a file path, a waveform (tensor or array; pass `sample_rate`), or a list of them.
            emotion: the requested emotion (Angry, Happy, Neutral, Sad, or Surprise), one per
                input or one for all. eMOS rates how closely the speech matches it, and the
                valence, arousal, and dominance heads also read it. If it is omitted, the
                category with the highest predicted probability is used, and `emotion_source`
                is "predicted".
            batch_size: utterances per forward pass.
            all_emotions: also return eMOS, valence, arousal, and dominance for each of the five
                requested emotions under "by_emotion", computed together in one pass.

        Returns:
            A dict per input with qmos, emos, valence, arousal, and dominance (1-5), category
            (probability that a listener selects each emotion category), emotion (the emotion
            used for eMOS), and emotion_source ("given" or "predicted").
        """
        single = not isinstance(audio, (list, tuple))
        items = [audio] if single else list(audio)
        if emotion is None or isinstance(emotion, str):
            emotions = [emotion] * len(items)
        else:
            emotions = list(emotion)
            if len(emotions) != len(items):
                raise ValueError(f"got {len(items)} inputs but {len(emotions)} emotions")
        names = [normalize_emotion(e) if e is not None else None for e in emotions]

        results: list[dict[str, Any]] = []
        for start in range(0, len(items), batch_size):
            waves = [load_audio(x, sample_rate, self.config.sample_rate) for x in items[start : start + batch_size]]
            results.extend(self._score_batch(waves, names[start : start + batch_size], listener_chunk_size, all_emotions))
        return results[0] if single else results


def load(
    name_or_path: str | os.PathLike = DEFAULT_REPO,
    device: str | torch.device | None = None,
    fp16: bool = True,
    revision: str | None = None,
) -> UTMOST:
    """Shortcut for `UTMOST.from_pretrained`."""
    return UTMOST.from_pretrained(name_or_path, device=device, fp16=fp16, revision=revision)

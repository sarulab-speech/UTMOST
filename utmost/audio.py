"""Audio loading and resampling."""

from __future__ import annotations

import math
import os
from typing import Any, Union

import torch
import torch.nn.functional as F
from torch import Tensor

AudioInput = Union[str, "os.PathLike[str]", Tensor, Any]

# One WavLM / wav2vec 2.0 frame needs 400 samples (25 ms) at 16 kHz.
MIN_SAMPLES = 400


def resample(waveform: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    """Band-limited sinc interpolation with a Hann window.

    Same computation as `torchaudio.functional.resample` with its default arguments
    (lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann").
    Adapted from torchaudio (BSD 2-Clause License); see THIRD_PARTY_NOTICES.md.
    """
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("sample rates must be positive")
    if int(orig_freq) != orig_freq or int(new_freq) != new_freq:
        raise ValueError("sample rates must be integers")
    if orig_freq == new_freq:
        return waveform
    lowpass_filter_width, rolloff = 6, 0.99
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // gcd, int(new_freq) // gcd
    dtype, device = waveform.dtype, waveform.device

    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)
    idx = torch.arange(-width, width + orig, dtype=dtype, device=device)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=dtype, device=device)[:, None, None] / new + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base_freq / orig)

    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    length = waveform.shape[-1]
    out = F.conv1d(F.pad(waveform, (width, width + orig))[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(waveform.shape[0], -1)
    target_length = int(torch.ceil(torch.as_tensor(new * length / orig)).long())
    return out[..., :target_length].reshape(shape[:-1] + (-1,))


def _to_float_tensor(audio: Any) -> Tensor:
    """Convert an array or tensor to float32; integer PCM is scaled to [-1, 1)."""
    wave = torch.as_tensor(audio)
    if wave.dtype == torch.uint8:
        return (wave.float() - 128.0) / 128.0
    if not wave.is_floating_point():
        bits = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}.get(wave.dtype)
        if bits is None:
            raise TypeError(f"unsupported waveform dtype {wave.dtype}")
        return wave.double().div(2.0 ** (bits - 1)).float()
    return wave.float()


def load_audio(audio: AudioInput, sample_rate: int | None = None, target_rate: int = 16000) -> Tensor:
    """Return a mono float32 waveform at `target_rate` Hz.

    `audio` is a file path (any format soundfile reads, e.g. WAV, FLAC, OGG, MP3) or a
    waveform of shape (samples,), (channels, samples), or (samples, channels) with its
    `sample_rate`. Multichannel audio is averaged over channels.
    """
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf

        try:
            data, sample_rate = sf.read(os.fspath(audio), dtype="float32", always_2d=True)
        except Exception as err:
            raise RuntimeError(f"could not read audio file {os.fspath(audio)!r}: {err}") from err
        wave = torch.from_numpy(data.T.copy())
    else:
        if sample_rate is None:
            raise ValueError("pass sample_rate when giving a waveform instead of a file path")
        wave = _to_float_tensor(audio)
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() == 2 and wave.shape[1] <= 16 < wave.shape[0]:
            wave = wave.T  # (samples, channels) -> (channels, samples)
    if wave.dim() != 2:
        raise ValueError(f"expected a mono or multichannel waveform, got shape {tuple(wave.shape)}")
    wave = wave.mean(dim=0)
    if sample_rate != target_rate:
        wave = resample(wave, int(sample_rate), target_rate)
    if wave.numel() < MIN_SAMPLES:
        raise ValueError(f"audio is too short ({wave.numel()} samples at {target_rate} Hz; at least {MIN_SAMPLES} needed)")
    return wave

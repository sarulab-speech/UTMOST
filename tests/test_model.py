"""Behaviour of the package with random weights (CPU, no download)."""

import numpy as np
import pytest
import soundfile as sf
import torch

from utmost import EMOTIONS, load_audio
from utmost.audio import resample
from utmost.layers import rename_weight_norm_keys


def test_single_and_batch(tiny_model):
    wave = torch.randn(24000) * 0.1
    one = tiny_model.predict(wave, emotion="happy", sample_rate=16000)
    assert one["emotion"] == "Happy" and one["emotion_source"] == "given"
    for key in ("qmos", "emos", "valence", "arousal", "dominance"):
        assert 1.0 <= one[key] <= 5.0
    assert set(one["category"]) == set(EMOTIONS)
    batch = tiny_model.predict([wave, wave[:8000]], emotion=["Happy", "Sad"], sample_rate=16000, batch_size=2)
    assert abs(batch[0]["qmos"] - one["qmos"]) < 1e-4  # padding does not change the longer item


def test_no_emotion_uses_most_likely_category(tiny_model):
    wave = torch.randn(16000) * 0.1
    out = tiny_model.predict(wave, sample_rate=16000)
    top = max(out["category"], key=out["category"].get)
    assert out["emotion"] == top and out["emotion_source"] == "predicted"
    given = tiny_model.predict(wave, emotion=top, sample_rate=16000)
    assert given["emos"] == out["emos"] and given["emotion_source"] == "given"


def test_qmos_and_category_ignore_the_requested_emotion(tiny_model):
    wave = torch.randn(16000) * 0.1
    a = tiny_model.predict(wave, emotion="Angry", sample_rate=16000)
    b = tiny_model.predict(wave, emotion="Sad", sample_rate=16000)
    assert a["qmos"] == b["qmos"] and a["category"] == b["category"]


def test_file_input_matches_array_input(tiny_model, tmp_path):
    wave = (np.random.default_rng(0).standard_normal(22050) * 0.1).astype(np.float32)
    path = tmp_path / "x.wav"
    sf.write(path, wave, 22050, subtype="FLOAT")
    a = tiny_model.predict(path, emotion="Neutral")
    b = tiny_model.predict(wave, emotion="Neutral", sample_rate=22050)
    assert abs(a["qmos"] - b["qmos"]) < 1e-6


def test_load_audio_formats():
    pcm = (np.random.default_rng(0).standard_normal(16000) * 3000).astype(np.int16)
    x = load_audio(pcm, sample_rate=16000)
    assert x.dtype == torch.float32 and x.abs().max() < 1.0
    stereo_last = np.stack([pcm, pcm], axis=1)  # (samples, channels)
    assert torch.allclose(load_audio(stereo_last, sample_rate=16000), x)
    assert load_audio(torch.randn(44100) * 0.1, sample_rate=44100).numel() == 16000
    with pytest.raises(ValueError):
        load_audio(torch.zeros(100), sample_rate=16000)


def test_resample_length():
    for sr in (8000, 22050, 24000, 44100, 48000):
        n = 12345
        assert resample(torch.randn(n), sr, 16000).numel() == int(np.ceil(16000 * n / sr))


def test_rename_weight_norm_keys():
    state = {"a.conv.parametrizations.weight.original0": 0, "a.conv.parametrizations.weight.original1": 1, "b": 2}
    assert rename_weight_norm_keys(state) == {"a.conv.weight_g": 0, "a.conv.weight_v": 1, "b": 2}


def test_all_emotions_matches_separate_calls(tiny_model):
    waves = [torch.randn(20000) * 0.1, torch.randn(12000) * 0.1]
    both = tiny_model.predict(waves, emotion=["Sad", None], sample_rate=16000, all_emotions=True, batch_size=2)
    plain = tiny_model.predict(waves, emotion=["Sad", None], sample_rate=16000, batch_size=2)
    for b, p in zip(both, plain):  # the usual fields do not change (up to float rounding)
        assert b["emotion"] == p["emotion"] and b["emotion_source"] == p["emotion_source"]
        for key in ("qmos", "emos", "valence", "arousal", "dominance"):
            assert abs(b[key] - p[key]) < 1e-5
        assert max(abs(b["category"][e] - p["category"][e]) for e in EMOTIONS) < 1e-5
        assert set(b["by_emotion"]) == set(EMOTIONS)
        assert abs(b["by_emotion"][b["emotion"]]["emos"] - p["emos"]) < 1e-5
    for name in EMOTIONS:
        single = tiny_model.predict(waves[0], emotion=name, sample_rate=16000)
        got = both[0]["by_emotion"][name]
        for key in ("emos", "valence", "arousal", "dominance"):
            assert abs(got[key] - single[key]) < 1e-5

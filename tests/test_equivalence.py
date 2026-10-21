"""The built-in WavLM and resampler match Transformers and torchaudio.

Skipped unless transformers / torchaudio are installed; they are not package dependencies.
"""

import pytest
import torch

from utmost.audio import resample
from utmost.layers import rename_weight_norm_keys
from utmost.wavlm import WavLM

from conftest import TINY_WAVLM


def test_resample_matches_torchaudio():
    torchaudio = pytest.importorskip("torchaudio")
    x = torch.randn(2, 12345)
    for sr in (8000, 22050, 24000, 44100, 48000):
        ref = torchaudio.functional.resample(x, sr, 16000)
        out = resample(x, sr, 16000)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-6, rtol=0)


def test_wavlm_matches_transformers():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.WavLMConfig(**TINY_WAVLM)
    reference = transformers.WavLMModel(config).eval()
    ours = WavLM(config.to_dict()).eval()
    ours.load_state_dict(rename_weight_norm_keys(reference.state_dict()), strict=True)

    x = torch.randn(2, 9000)
    x[1, 6000:] = 0.0
    mask = torch.ones(2, 9000, dtype=torch.long)
    mask[1, 6000:] = 0
    with torch.no_grad():
        expected = reference(x, attention_mask=mask).last_hidden_state
        got = ours(x, mask)
    valid = ours.output_lengths(mask.sum(dim=-1))
    for i in range(2):
        assert torch.allclose(got[i, : valid[i]], expected[i, : valid[i]], atol=1e-5, rtol=1e-4)

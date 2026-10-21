import pytest
import torch

from utmost import UTMOST
from utmost.model import UTMOSTConfig

# A small WavLM configuration with the same structure as WavLM-Large.
TINY_WAVLM = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": [32, 32],
    "conv_kernel": [10, 3],
    "conv_stride": [5, 2],
    "conv_bias": False,
    "num_conv_pos_embeddings": 16,
    "num_conv_pos_embedding_groups": 2,
    "num_buckets": 320,
    "max_bucket_distance": 800,
    "layer_norm_eps": 1e-5,
    "feat_extract_norm": "layer",
    "do_stable_layer_norm": True,
}


@pytest.fixture(scope="session")
def tiny_model() -> UTMOST:
    """UTMOST with random weights and a small WavLM; no download needed."""
    torch.manual_seed(0)
    config = UTMOSTConfig(wavlm=TINY_WAVLM, num_listeners=10, listener_ids=[0, 3, 6, 9], max_duration_sec=1.0)
    model = UTMOST(config).eval()
    for p in model.network.utmos22.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return model

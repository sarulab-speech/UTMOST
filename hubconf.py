"""torch.hub entry point: torch.hub.load("sarulab-speech/UTMOST", "utmost")."""

dependencies = ["torch", "soundfile", "safetensors", "huggingface_hub"]

from utmost.api import DEFAULT_REPO, UTMOST  # noqa: E402


def utmost(pretrained: bool = True, device=None, fp16: bool = True, weights: str = DEFAULT_REPO) -> UTMOST:
    """UTMOST with pretrained weights from the Hugging Face Hub (or a local directory via `weights`).

    >>> model = torch.hub.load("sarulab-speech/UTMOST", "utmost", trust_repo=True)
    >>> model.predict("sample.wav", emotion="Happy")
    """
    if not pretrained:
        raise ValueError("UTMOST is only distributed with pretrained weights")
    return UTMOST.from_pretrained(weights, device=device, fp16=fp16)

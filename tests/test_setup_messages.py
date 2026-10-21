"""Messages for common setup problems: no access to the weights, and PyTorch without CUDA."""

import sys
import types
import warnings

import pytest
import torch

from utmost import api


class RepositoryNotFoundError(Exception):
    """Stands in for huggingface_hub's error for a private repository without login."""

    response = types.SimpleNamespace(status_code=401)


def test_private_repository_error_explains_login(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")

    def fail(*args, **kwargs):
        raise RepositoryNotFoundError("401 Unauthorized")

    monkeypatch.setattr(hub, "hf_hub_download", fail)
    with pytest.raises(RuntimeError, match="not public yet.*hf auth login"):
        api._resolve_files("sarulab-speech/UTMOST", "fp16", None)
    with pytest.raises(RuntimeError, match="other/repo.*hf auth login"):
        api._resolve_files("other/repo", "fp16", None)


def test_warns_when_gpu_present_but_torch_has_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(api.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.warns(UserWarning, match="download.pytorch.org/whl/cu130"):
        api._warn_if_gpu_unused()


def test_no_warning_without_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(api.shutil, "which", lambda name: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        api._warn_if_gpu_unused()

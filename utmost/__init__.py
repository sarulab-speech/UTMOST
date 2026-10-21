"""UTMOST: a multi-task model for assessing emotional speech synthesis."""

from .api import UTMOST, load
from .audio import load_audio
from .model import EMOTIONS

__version__ = "0.1.0"
__all__ = ["UTMOST", "load", "load_audio", "EMOTIONS", "__version__"]

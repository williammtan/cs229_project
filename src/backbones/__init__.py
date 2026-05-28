"""Backbones: raw-EEG → class probabilities (monolithic) or raw-EEG → embedding (FM)."""
from src.backbones.base import BackboneBase

# Side-effect: register backbones into src.core.registry on import.
from src.backbones import csp_lda  # noqa: F401
from src.backbones import riemann_lr  # noqa: F401
from src.backbones import eegnet  # noqa: F401
from src.backbones import shallowconvnet  # noqa: F401
from src.backbones import cbramod  # noqa: F401
from src.backbones import labram  # noqa: F401
from src.backbones import reve  # noqa: F401

__all__ = ["BackboneBase"]

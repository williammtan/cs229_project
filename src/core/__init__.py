"""Core scaffolding: registry, pipeline composition, shared types."""
from src.core.registry import register, get_registry, build_from_cfg
from src.core.types import Split, RunMeta
from src.core.pipeline import Pipeline

__all__ = ["register", "get_registry", "build_from_cfg", "Split", "RunMeta", "Pipeline"]

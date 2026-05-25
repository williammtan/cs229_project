"""Per-subject adapters. Composed via AdapterStack in execution order."""
from src.adapters.base import AdapterBase, NoopAdapter
from src.adapters.composite import AdapterStack

# Register the noop for Hydra reference.
from src.adapters import _register_builtins  # noqa: F401

__all__ = ["AdapterBase", "NoopAdapter", "AdapterStack"]

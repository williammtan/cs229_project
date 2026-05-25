"""Per-subject adapters. Composed via AdapterStack in execution order."""
from src.adapters.base import AdapterBase, NoopAdapter
from src.adapters.composite import AdapterStack

# Side-effect: register adapters into src.core.registry on import.
from src.adapters import _register_builtins  # noqa: F401
from src.adapters import riemannian  # noqa: F401  registers ra_static, ra_ema

__all__ = ["AdapterBase", "NoopAdapter", "AdapterStack"]

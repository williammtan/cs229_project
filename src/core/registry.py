"""Light registry for plug-in classes (backbones, adapters, heads, protocols).

We could lean entirely on Hydra's ``_target_`` instantiation, but a tiny registry
buys us discoverability (``list(get_registry("backbone").keys())``) and lets
configs reference short names instead of dotted paths.
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = {}


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: ``@register("backbone", "eegnet")`` adds the class to the registry.

    Raises if the name is already taken in that kind (no silent overwrites).
    """
    bucket = _REGISTRIES.setdefault(kind, {})

    def deco(cls: type[T]) -> type[T]:
        if name in bucket:
            raise KeyError(f"{kind!r} already has a class registered as {name!r}: {bucket[name]}")
        bucket[name] = cls
        return cls

    return deco


def get_registry(kind: str) -> dict[str, type]:
    return _REGISTRIES.get(kind, {})


def build_from_cfg(kind: str, cfg: dict[str, Any]) -> Any:
    """Instantiate ``cfg["name"]`` from the ``kind`` registry, passing remaining
    fields as kwargs. ``cfg`` is typically an OmegaConf node converted via
    ``OmegaConf.to_container(node, resolve=True)``.
    """
    if "name" not in cfg:
        raise ValueError(f"{kind} config missing 'name' field: {cfg}")
    name = cfg["name"]
    bucket = get_registry(kind)
    if name not in bucket:
        raise KeyError(f"{kind!r} {name!r} not registered. Known: {sorted(bucket)}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return bucket[name](**kwargs)

"""cx-pangram: local EditLens AI-edit detection (engine imported lazily)."""

from __future__ import annotations

__all__ = [
    "EditLens",
    "Detection",
    "ChunkScore",
    "ModelAccessError",
    "band_for",
    "get_engine",
]


class ModelAccessError(RuntimeError):
    """An HF repo the engine needs is unreachable: gated, missing, or offline-uncached.

    Defined here (not in :mod:`.engine`) so CLI surfaces can catch it without
    paying the torch import.
    """


_LAZY = ("EditLens", "Detection", "ChunkScore", "band_for", "get_engine")


def __getattr__(name: str):
    if name in _LAZY:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

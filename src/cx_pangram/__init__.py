"""cx-pangram: local EditLens AI-edit detection (engine imported lazily)."""

from __future__ import annotations

__all__ = ["EditLens", "Detection", "ChunkScore", "band_for"]


def __getattr__(name: str):
    if name in __all__:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

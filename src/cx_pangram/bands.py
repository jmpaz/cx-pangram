"""Score bands: the one place thresholds, labels, and display colors live together.

Both the torch-heavy engine and the light lens/formatter layer need the band
table, and the import barrier between them (lens must stay importable without
torch) is exactly how two copies drifted apart historically. This module is
dependency-free so every layer can share it.

The default cuts were calibrated to the llama backbone (see the eval harness's
``propose_bands``); ``from_artifact`` loads recalibrated cuts written by
``cx-pangram eval --write-bands``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    hi: float  # exclusive upper bound on the score
    label: str
    color: str  # rich color name


_COLORS = ("green", "yellow", "dark_orange", "red", "red")

BANDS: tuple[Band, ...] = (
    Band(0.30, "human", "green"),
    Band(0.55, "lightly edited", "yellow"),
    Band(0.75, "moderately edited", "dark_orange"),
    Band(0.90, "heavily edited", "red"),
    Band(float("inf"), "fully AI", "red"),
)


def band_for(score: float, bands: tuple[Band, ...] = BANDS) -> Band:
    for band in bands:
        if score < band.hi:
            return band
    return bands[-1]


def from_artifact(artifact: dict) -> tuple[Band, ...]:
    """Build a band table from a ``cx-pangram eval --write-bands`` artifact.

    Colors are assigned positionally from the default palette — the artifact
    records cuts and provenance, not presentation.
    """
    if artifact.get("kind") != "cx-pangram-bands":
        raise ValueError("not a cx-pangram bands artifact (missing kind)")
    if artifact.get("version") != 1:
        raise ValueError(
            f"unsupported bands artifact version {artifact.get('version')!r}"
        )
    if artifact.get("degenerate"):
        raise ValueError(
            "bands artifact is flagged degenerate (collapsed span); refusing to load"
        )
    cuts = artifact["cuts"]
    if len(cuts) != len(BANDS) - 1:
        raise ValueError(f"expected {len(BANDS) - 1} cuts, got {len(cuts)}")
    bands = [
        Band(float(c["lt"]), str(c["band"]), _COLORS[i]) for i, c in enumerate(cuts)
    ]
    lows = [b.hi for b in bands]
    if lows != sorted(lows):
        raise ValueError("bands artifact cuts are not monotonically non-decreasing")
    top_label = str(artifact.get("top_band", BANDS[-1].label))
    bands.append(Band(float("inf"), top_label, _COLORS[-1]))
    return tuple(bands)

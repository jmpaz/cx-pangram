"""The shared band table and calibration-artifact round-trip."""

import pytest

from cx_pangram.bands import BANDS, Band, band_for, from_artifact


def test_default_bands_are_monotone_and_labeled():
    cuts = [b.hi for b in BANDS]
    assert cuts == sorted(cuts)
    assert BANDS[0].label == "human" and BANDS[-1].label == "fully AI"


def test_band_for_maps_scores_to_bands():
    assert band_for(0.0).label == "human"
    assert band_for(0.29).label == "human"
    assert band_for(0.30).label == "lightly edited"
    assert band_for(0.30).color == "yellow"
    assert band_for(0.95).label == "fully AI"
    assert band_for(1.0).label == "fully AI"


def test_band_for_accepts_custom_table():
    custom = (Band(0.5, "low", "green"), Band(float("inf"), "high", "red"))
    assert band_for(0.4, custom).label == "low"
    assert band_for(0.6, custom).label == "high"


def _artifact(**overrides):
    art = {
        "version": 1,
        "kind": "cx-pangram-bands",
        "model": "llama",
        "cuts": [
            {"lt": 0.2, "band": "human", "pinned": True},
            {"lt": 0.4, "band": "lightly edited", "pinned": True},
            {"lt": 0.6, "band": "moderately edited", "pinned": False},
            {"lt": 0.8, "band": "heavily edited", "pinned": True},
        ],
        "top_band": "fully AI",
        "degenerate": False,
    }
    art.update(overrides)
    return art


def test_from_artifact_round_trips():
    bands = from_artifact(_artifact())
    assert len(bands) == 5
    assert band_for(0.1, bands).label == "human"
    assert band_for(0.7, bands).label == "heavily edited"
    assert band_for(0.9, bands).label == "fully AI"
    assert all(b.color for b in bands)


def test_from_artifact_rejects_bad_input():
    with pytest.raises(ValueError, match="kind"):
        from_artifact({"version": 1})
    with pytest.raises(ValueError, match="version"):
        from_artifact(_artifact(version=2))
    with pytest.raises(ValueError, match="degenerate"):
        from_artifact(_artifact(degenerate=True))
    bad = _artifact()
    bad["cuts"][1]["lt"] = 0.1
    with pytest.raises(ValueError, match="monotonic"):
        from_artifact(bad)
    with pytest.raises(ValueError, match="expected 4 cuts"):
        from_artifact(_artifact(cuts=[{"lt": 0.5, "band": "human", "pinned": True}]))

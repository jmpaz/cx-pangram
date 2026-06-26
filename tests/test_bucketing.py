"""score_to_bucket boundary contract: lo/hi thresholds and even interior split."""

import pytest

from cx_pangram.eval import score_to_bucket


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, 0),
        (0.029, 0),
        (0.03, 0),
        (0.05, 1),
        (0.089, 1),
        (0.09, 2),
        (0.149, 2),
        (0.15, 3),
        (0.30, 3),
        (1.0, 3),
    ],
)
def test_default_boundaries(score, expected):
    assert score_to_bucket(score) == expected


def test_lo_threshold_is_closed_below():
    assert score_to_bucket(0.03 - 1e-9) == 0
    assert score_to_bucket(0.03) == 0


def test_hi_threshold_is_closed_above():
    assert score_to_bucket(0.15 - 1e-9) == 2
    assert score_to_bucket(0.15) == 3


def test_interior_midpoint_splits_evenly():
    mid = (0.03 + 0.15) / 2
    assert score_to_bucket(mid - 1e-6) == 1
    assert score_to_bucket(mid) == 2


def test_custom_thresholds_and_bucket_count():
    assert score_to_bucket(0.05, lo_th=0.1, hi_th=0.9, n_buckets=3) == 0
    assert score_to_bucket(0.5, lo_th=0.1, hi_th=0.9, n_buckets=3) == 1
    assert score_to_bucket(0.95, lo_th=0.1, hi_th=0.9, n_buckets=3) == 2


def test_binary_buckets_split_at_lo_threshold():
    assert score_to_bucket(0.05, lo_th=0.1, hi_th=0.9, n_buckets=2) == 0
    assert score_to_bucket(0.5, lo_th=0.1, hi_th=0.9, n_buckets=2) == 1
    assert score_to_bucket(0.95, lo_th=0.1, hi_th=0.9, n_buckets=2) == 1

"""numpy metric helpers, cross-checked against numpy/scipy where available."""

import numpy as np
import pytest

from cx_pangram.eval import (
    _balacc_sweep,
    _confusion,
    _describe,
    _fp_table,
    _rankdata,
    _spearman,
)


def test_rankdata_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=40).astype(float)
    np.testing.assert_allclose(_rankdata(a), scipy_stats.rankdata(a))


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = 0.7 * x + rng.normal(size=60)
    expected = scipy_stats.spearmanr(x, y).statistic
    assert _spearman(x, y) == pytest.approx(expected, abs=1e-9)


def test_spearman_monotone_invariant_to_scale():
    x = np.arange(1.0, 11.0)
    y = 3.0 * x + 5.0
    assert _spearman(x, y) == pytest.approx(1.0)
    assert _spearman(x, -y) == pytest.approx(-1.0)


def test_spearman_short_and_constant_are_nan():
    assert np.isnan(_spearman([1.0], [2.0]))
    assert np.isnan(_spearman([1.0, 1.0, 1.0], [3.0, 4.0, 5.0]))


def test_confusion_rows_are_ground_truth():
    gt = [0, 0, 1, 2, 3, 3]
    pred = [0, 1, 1, 2, 3, 2]
    m = _confusion(gt, pred)
    assert m.shape == (4, 4)
    assert m[0, 0] == 1 and m[0, 1] == 1
    assert m[1, 1] == 1
    assert m[3, 3] == 1 and m[3, 2] == 1
    assert m.sum() == len(gt)


def test_describe_matches_numpy():
    xs = [0.1, 0.2, 0.2, 0.4, 0.9]
    d = _describe(xs)
    assert d["n"] == 5
    assert d["min"] == pytest.approx(0.1)
    assert d["max"] == pytest.approx(0.9)
    assert d["median"] == pytest.approx(float(np.median(xs)))
    assert d["mean"] == pytest.approx(float(np.mean(xs)))
    assert d["q25"] == pytest.approx(float(np.percentile(xs, 25)))
    assert d["q90"] == pytest.approx(float(np.percentile(xs, 90)))


def test_describe_empty_is_nan():
    d = _describe([])
    assert d["n"] == 0
    assert np.isnan(d["mean"])


def test_fp_table_counts_at_or_above_threshold():
    xs = [0.05, 0.10, 0.41, 0.5, 0.95]
    table = dict(_fp_table(xs))
    assert table[0.10] == 4
    assert table[0.40] == 3
    assert table[0.50] == 2
    assert table[0.90] == 1


def test_balacc_sweep_separates_clean_classes():
    neg = [0.0, 0.1, 0.2]
    pos = [0.8, 0.9, 1.0]
    best = _balacc_sweep(neg, pos)
    assert best is not None
    assert best["bal_acc"] == pytest.approx(1.0)
    assert 0.2 < best["threshold"] <= 0.8
    assert best["fp"] == 0


def test_balacc_sweep_empty_class_is_none():
    assert _balacc_sweep([], [0.5]) is None
    assert _balacc_sweep([0.5], []) is None

"""compute_dataset_metrics and propose_bands over synthetic Scored fixtures."""

import numpy as np
import pytest

from cx_pangram.eval import Scored, compute_dataset_metrics, propose_bands


def _scored(gt_bucket, model_bucket, model_score, cosine_score, *, reliable=True):
    return Scored(
        preview="x",
        n_words=100,
        reliable=reliable,
        cosine_score=cosine_score,
        gt_bucket=gt_bucket,
        model_score=model_score,
        model_bucket=model_bucket,
        band="band",
    )


def _gradient(n_per=6, jitter=0.0):
    rng = np.random.default_rng(0)
    rows = []
    centers = {0: 0.05, 1: 0.30, 2: 0.60, 3: 0.92}
    cosines = {0: 0.01, 1: 0.06, 2: 0.12, 3: 0.30}
    for b in range(4):
        for _ in range(n_per):
            s = float(np.clip(centers[b] + rng.normal(0, jitter), 0, 1))
            mb = int(round(s * 3))
            rows.append(_scored(b, mb, s, cosines[b]))
    return rows


def test_perfect_agreement_metrics():
    rows = [
        _scored(0, 0, 0.05, 0.01),
        _scored(1, 1, 0.33, 0.06),
        _scored(2, 2, 0.66, 0.12),
        _scored(3, 3, 0.95, 0.30),
    ]
    m = compute_dataset_metrics(rows)
    assert m.n == 4
    assert m.bucket_accuracy == pytest.approx(1.0)
    assert m.adjacent_accuracy == pytest.approx(1.0)
    assert m.bucket_mae == pytest.approx(0.0)
    assert m.spearman_bucket == pytest.approx(1.0)
    assert m.spearman_score == pytest.approx(1.0)
    assert m.confusion[0][0] == 1 and m.confusion[3][3] == 1


def test_off_by_one_drives_adjacent_and_mae():
    rows = [
        _scored(0, 0, 0.05, 0.01),
        _scored(1, 2, 0.66, 0.06),
        _scored(2, 2, 0.66, 0.12),
        _scored(3, 3, 0.95, 0.30),
    ]
    m = compute_dataset_metrics(rows)
    assert m.bucket_accuracy == pytest.approx(0.75)
    assert m.adjacent_accuracy == pytest.approx(1.0)
    assert m.bucket_mae == pytest.approx(0.25)


def test_per_gt_score_groups_model_scores():
    m = compute_dataset_metrics(_gradient())
    assert set(m.per_gt_score) == {0, 1, 2, 3}
    means = [m.per_gt_score[b]["mean"] for b in range(4)]
    assert means == sorted(means)


def test_empty_metrics_are_nan_safe():
    m = compute_dataset_metrics([])
    assert m.n == 0
    assert np.isnan(m.bucket_accuracy)
    assert len(m.confusion) == 4 and len(m.confusion[0]) == 4


def test_source_control_reads_human():
    rows = _gradient()
    for r in rows:
        r.source_score = 0.02
        r.source_bucket = 0
    m = compute_dataset_metrics(rows, with_source=True)
    assert m.source_control is not None
    assert m.source_control["frac_bucket0"] == pytest.approx(1.0)


def test_metrics_to_dict_stringifies_per_gt_keys():
    d = compute_dataset_metrics(_gradient()).to_dict()
    assert set(d["per_gt_score"]) == {"0", "1", "2", "3"}
    assert d["confusion"] and "spearman_score" in d


def test_propose_bands_pins_endpoints_and_interpolates_interior():
    prop = propose_bands(_gradient())
    cuts = [c for c, _, _ in prop.boundaries]
    pinned = [p for _, _, p in prop.boundaries]
    names = [name for _, name, _ in prop.boundaries]
    assert names == ["human", "lightly edited", "moderately edited", "heavily edited"]
    assert pinned == [True, False, False, True]
    assert cuts == sorted(cuts)
    assert prop.sweep_human is not None and prop.sweep_ai is not None
    span = cuts[-1] - cuts[0]
    assert cuts[1] == pytest.approx(cuts[0] + span / 3, abs=1e-3)
    assert cuts[2] == pytest.approx(cuts[0] + 2 * span / 3, abs=1e-3)

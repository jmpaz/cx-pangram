"""_compare_runs pairs two engine configs over identical samples with one filter."""

from types import SimpleNamespace

import pytest

import cx_pangram.engine as engine_mod
from cx_pangram.eval import Sample, _compare_runs


class _FakeEngine:
    """Score-by-lookup engine; texts under 50 words read as unreliable."""

    n_buckets = 4

    def __init__(self, scores, model_name, **kw):
        self._scores = scores
        self.model_name = model_name
        self.device = "cpu"
        self.quantized = bool(kw.get("quantize"))

    def detect(self, text):
        score = self._scores[text]
        n_words = len(text.split())
        return SimpleNamespace(
            score=score,
            bucket=round(score * 3),
            n_words=n_words,
            reliable=n_words >= 50,
            chunks=[],
        )


def _samples():
    long_a = "word " * 60
    long_b = "term " * 80
    short = "tiny text"
    return [
        Sample(text=long_a, cosine_score=0.01, gt_bucket=0, n_words=60),
        Sample(text=short, cosine_score=0.20, gt_bucket=3, n_words=2),
        Sample(text=long_b, cosine_score=0.16, gt_bucket=3, n_words=80),
    ]


@pytest.fixture
def fake_engines(monkeypatch):
    scores_by_label = {
        "a": {s.text: v for s, v in zip(_samples(), [0.05, 0.5, 0.9], strict=True)},
        "b": {s.text: v for s, v in zip(_samples(), [0.10, 0.5, 0.7], strict=True)},
    }
    labels = iter(["a", "b"])

    def factory(**kw):
        label = next(labels)
        return _FakeEngine(scores_by_label[label], model_name=f"fake-{label}", **kw)

    monkeypatch.setattr(engine_mod, "EditLens", factory)


def test_reliable_filter_applies_to_both_runs(fake_engines):
    cmp = _compare_runs(
        _samples(), {}, {}, ("a", "b"), reliable_only=True, with_source=False
    )
    assert cmp.n == 2
    assert cmp.metrics_a.n == cmp.metrics_b.n == 2
    assert cmp.score_mae == pytest.approx((0.05 + 0.2) / 2)
    assert cmp.max_delta == pytest.approx(0.2)


def test_all_rows_kept_without_filter(fake_engines):
    cmp = _compare_runs(
        _samples(), {}, {}, ("a", "b"), reliable_only=False, with_source=False
    )
    assert cmp.n == 3
    assert cmp.metrics_a.n == 3
    assert cmp.label_a == "a" and cmp.label_b == "b"
    assert cmp.latency_a["p50_s"] >= 0.0

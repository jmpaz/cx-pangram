"""Window batching, VRAM-derived batch width, and OOM backoff — no model weights.

Engines are built with ``object.__new__`` and a stubbed ``_forward`` so the
batching seam in ``_score`` / ``_auto_batch`` / ``detect`` is exercised alone.
"""

from types import SimpleNamespace

import pytest
import torch

from cx_pangram.engine import (
    CHUNK_WORDS,
    DEFAULT_BATCH,
    MAX_BATCH,
    EditLens,
)


def _bare_engine(device="cpu", n_buckets=4):
    eng = object.__new__(EditLens)
    eng.device = device
    eng.model_name = "stub"
    eng.calibrated = False
    eng.n_buckets = n_buckets
    eng.max_length = 512
    return eng


def _linear_forward(record=None):
    """Deterministic per-text scores: length-derived, order-preserving."""

    def forward(texts):
        if record is not None:
            record.append(len(texts))
        scores = [min(1.0, len(t) / 10000.0) for t in texts]
        buckets = [round(s * 3) for s in scores]
        probs = [[1.0 - s, 0.0, 0.0, s] for s in scores]
        return scores, buckets, probs

    return forward


def test_score_preserves_order_across_batches(monkeypatch):
    eng = _bare_engine()
    widths = []
    monkeypatch.setattr(eng, "_forward", _linear_forward(widths), raising=False)
    texts = [f"{'x' * (i + 1)}" for i in range(20)]
    scores, buckets, probs = eng._score(texts)
    assert len(scores) == len(buckets) == len(probs) == 20
    assert scores == sorted(scores)
    assert all(w <= DEFAULT_BATCH for w in widths)
    assert sum(w for w in widths) == 20


def test_oom_backoff_halves_and_recovers(monkeypatch):
    eng = _bare_engine()
    inner = _linear_forward()
    widths = []

    def flaky(texts):
        widths.append(len(texts))
        if len(texts) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")
        return inner(texts)

    monkeypatch.setattr(eng, "_forward", flaky, raising=False)
    texts = [f"{'x' * (i + 1)}" for i in range(8)]
    scores, _, _ = eng._score(texts)
    assert len(scores) == 8
    assert scores == sorted(scores)
    assert max(widths) > 2
    assert all(w <= 2 for w in widths[widths.index(max(widths)) + 2 :])


def test_oom_at_batch_one_reraises(monkeypatch):
    eng = _bare_engine()

    def always_oom(texts):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (stub)")

    monkeypatch.setattr(eng, "_forward", always_oom, raising=False)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        eng._score(["a", "b"])


def test_auto_batch_cpu_uses_default_width():
    eng = _bare_engine(device="cpu")
    assert eng._auto_batch(3) == 3
    assert eng._auto_batch(100) == DEFAULT_BATCH


def test_auto_batch_cuda_scales_with_free_vram(monkeypatch):
    eng = _bare_engine(device="cuda:0")
    eng.net = SimpleNamespace(
        config=SimpleNamespace(intermediate_size=8192, hidden_size=2048)
    )
    eng.max_length = 1024

    per_window = eng.max_length * 8192 * 2 * 24  # mirrors ACT_SAFETY formula
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev: (per_window * 8 * 2, per_window * 32)
    )
    assert eng._auto_batch(100) == 8

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (0, per_window))
    assert eng._auto_batch(100) == 1

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda dev: (per_window * MAX_BATCH * 10, per_window * MAX_BATCH * 20),
    )
    assert eng._auto_batch(4096) == MAX_BATCH


def test_detect_windows_long_text_and_aggregates(monkeypatch):
    eng = _bare_engine()
    monkeypatch.setattr(eng, "_forward", _linear_forward(), raising=False)
    words = ["alpha"] * (CHUNK_WORDS * 2 + 25)
    det = eng.detect(" ".join(words))
    assert det.n_chunks == 3
    assert [c.n_words for c in det.chunks] == [CHUNK_WORDS, CHUNK_WORDS, 25]
    assert det.n_words >= sum(c.n_words for c in det.chunks)
    weighted = sum(c.score * c.n_words for c in det.chunks) / sum(
        c.n_words for c in det.chunks
    )
    assert det.score == pytest.approx(weighted, abs=1e-4)
    assert det.most_ai_chunk == max(det.chunks, key=lambda c: c.score).index


def test_detect_result_invariant_to_batch_width(monkeypatch):
    words = ["beta"] * (CHUNK_WORDS * 4)
    text = " ".join(words)

    def run(width):
        eng = _bare_engine()
        monkeypatch.setattr(eng, "_forward", _linear_forward(), raising=False)
        monkeypatch.setattr(eng, "_auto_batch", lambda n: width, raising=False)
        return eng.detect(text)

    one = run(1)
    four = run(4)
    assert [c.score for c in one.chunks] == [c.score for c in four.chunks]
    assert one.score == four.score

"""load_sample over a canned streaming-dataset seam (no network / no gated access)."""

import importlib.util
import sys
import types

import pytest

import cx_pangram.eval as ev


class _FakeStream:
    def __init__(self, rows):
        self._rows = rows
        self.shuffle_calls = []
        self.take_n = None

    def shuffle(self, seed=None, buffer_size=None):
        self.shuffle_calls.append((seed, buffer_size))
        return self

    def take(self, n):
        self.take_n = n
        return self

    def __iter__(self):
        rows = self._rows if self.take_n is None else self._rows[: self.take_n]
        return iter(rows)


def _install_fake_datasets(monkeypatch, rows, capture=None):
    stream = _FakeStream(rows)

    def load_dataset(dataset_id, split=None, streaming=False):
        if capture is not None:
            capture.update(
                {"id": dataset_id, "split": split, "streaming": streaming}
            )
        return stream

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return stream


def _rows():
    return [
        {"text": "word " * 80, "cosine_score": 0.01, "source_text": "src one"},
        {"text": "word " * 80, "cosine_score": 0.07, "source_text": "src two"},
        {"text": "tiny text", "cosine_score": 0.20, "source_text": None},
    ]


def test_load_sample_builds_samples(monkeypatch):
    capture = {}
    _install_fake_datasets(monkeypatch, _rows(), capture)
    samples = ev.load_sample(3, split="test", seed=7)
    assert capture == {"id": ev.DATASET_ID, "split": "test", "streaming": True}
    assert len(samples) == 3
    assert [s.gt_bucket for s in samples] == [0, 1, 3]
    assert samples[0].n_words == 80
    assert samples[2].n_words == 2
    assert samples[0].source_text == "src one"


def test_load_sample_passes_seed_and_buffer(monkeypatch):
    stream = _install_fake_datasets(monkeypatch, _rows())
    ev.load_sample(2, seed=42, buffer_size=1234)
    assert stream.shuffle_calls == [(42, 1234)]
    assert stream.take_n == 2


def test_load_sample_take_limits_rows(monkeypatch):
    _install_fake_datasets(monkeypatch, _rows())
    samples = ev.load_sample(2)
    assert len(samples) == 2


def test_load_sample_missing_datasets_is_actionable(monkeypatch):
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets is installed; cannot exercise the missing path")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        ev.load_sample(1)
    assert "datasets" in str(excinfo.value) or "cx-pangram[eval]" in str(excinfo.value)

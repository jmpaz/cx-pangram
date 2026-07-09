"""score_targets partitions stdin/file/ref targets and batches them through one engine."""

import io
from types import SimpleNamespace

import pytest

import cx_pangram.engine as engine_mod
import cx_pangram.lens as lens


class _FakeEngine:
    def __init__(self):
        self.batches = []

    def detect_batch(self, texts):
        self.batches.append(list(texts))
        out = []
        for t in texts:
            n = len(t.split())
            out.append(
                SimpleNamespace(
                    score=0.5,
                    band="lightly edited",
                    bucket=1,
                    n_words=n,
                    n_chunks=1,
                    reliable=n >= 50,
                    confidence=0.8,
                    truncated=False,
                    model="fake",
                    calibrated=True,
                    most_ai_chunk=0,
                    chunks=[],
                )
            )
        return out


@pytest.fixture
def fake_engine(monkeypatch):
    eng = _FakeEngine()
    monkeypatch.setattr(engine_mod, "get_engine", lambda **kw: eng)
    return eng


def test_files_and_stdin_score_without_contextualize(
    fake_engine, monkeypatch, tmp_path
):
    f = tmp_path / "doc.txt"
    f.write_text("word " * 60)
    monkeypatch.setattr(lens.sys, "stdin", io.StringIO("piped " * 55))

    results = lens.score_targets([str(f), "-"])
    assert [r["label"] for r in results] == [str(f), "(stdin)"]
    assert all(not r["skipped"] for r in results)
    assert len(fake_engine.batches) == 1 and len(fake_engine.batches[0]) == 2


def test_ref_targets_resolve_through_contextualize(fake_engine, monkeypatch, tmp_path):
    import sys as _sys

    doc = SimpleNamespace(
        source="https://x.test/a", label="a-doc", prose="prose " * 60, metadata={}
    )
    fake_ctx = SimpleNamespace(resolve_refs=lambda refs, **kw: [doc])
    monkeypatch.setitem(_sys.modules, "contextualize", fake_ctx)

    f = tmp_path / "local.txt"
    f.write_text("word " * 60)
    results = lens.score_targets([str(f), "https://x.test/a"])
    assert [r["label"] for r in results] == [str(f), "a-doc"]
    assert results[1]["source"] == "https://x.test/a"
    assert len(fake_engine.batches[0]) == 2


def test_missing_path_is_treated_as_ref(fake_engine, monkeypatch):
    import sys as _sys

    seen = []

    def resolve_refs(refs, **kw):
        seen.extend(refs)
        return []

    monkeypatch.setitem(
        _sys.modules, "contextualize", SimpleNamespace(resolve_refs=resolve_refs)
    )
    results = lens.score_targets(["definitely/not/a/file.txt"])
    assert seen == ["definitely/not/a/file.txt"]
    assert results == []

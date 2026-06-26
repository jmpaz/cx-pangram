"""eval_refs over a canned score_refs seam (no contextualize / no engine)."""

import cx_pangram.eval as ev
import cx_pangram.lens as lens


def _entry(label, source, score, *, reliable=True, skipped=False, reason=None):
    return {
        "ref": source,
        "source": source,
        "label": label,
        "score": score,
        "band": "human" if (score or 0) < 0.3 else "edited",
        "bucket": 0,
        "n_words": 120 if reliable else 10,
        "n_chunks": 1,
        "reliable": reliable,
        "model": "roberta",
        "calibrated": False,
        "skipped": skipped,
        "reason": reason,
        "preview": label,
    }


def _patch(monkeypatch, entries):
    monkeypatch.setattr(lens, "score_refs", lambda targets, **kw: list(entries))


def test_eval_refs_groups_and_describes(monkeypatch):
    entries = [
        _entry("a", "chan/a", 0.10),
        _entry("b", "chan/b", 0.20),
        _entry("c", "chan/c", 0.55),
        _entry("d", "skipme", None, skipped=True, reason="no prose"),
    ]
    _patch(monkeypatch, entries)
    report = ev.eval_refs(["chan"])
    assert report.n_scored == 3
    assert report.n_skipped == 1
    assert report.overall["n"] == 3
    assert report.overall["max"] == 0.55
    table = dict(report.fp_table)
    assert table[0.50] == 1
    assert table[0.10] == 3
    assert report.skipped[0]["reason"] == "no prose"
    assert report.model == "roberta"


def test_eval_refs_reliable_only_filter(monkeypatch):
    entries = [
        _entry("long", "s1", 0.40, reliable=True),
        _entry("short", "s2", 0.90, reliable=False),
    ]
    _patch(monkeypatch, entries)
    rel = ev.eval_refs(["x"], reliable_only=True)
    assert rel.overall["n"] == 1
    assert rel.overall["max"] == 0.40
    every = ev.eval_refs(["x"], reliable_only=False)
    assert every.overall["n"] == 2


def test_eval_refs_by_source_when_multiple(monkeypatch):
    entries = [
        _entry("a", "src-one", 0.1),
        _entry("b", "src-one", 0.3),
        _entry("c", "src-two", 0.7),
    ]
    _patch(monkeypatch, entries)
    report = ev.eval_refs(["x"])
    assert set(report.by_source) == {"src-one", "src-two"}
    assert report.by_source["src-one"]["n"] == 2


def test_eval_refs_passes_quantize_through(monkeypatch):
    seen = {}

    def fake(targets, **kw):
        seen.update(kw)
        return [_entry("a", "s", 0.1)]

    monkeypatch.setattr(lens, "score_refs", fake)
    ev.eval_refs(["x"], quantize=False, model="roberta")
    assert seen["quantize"] is False
    assert seen["model"] == "roberta"

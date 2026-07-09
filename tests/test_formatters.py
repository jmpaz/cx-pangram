"""Formatters render report fixtures to rich strings and json-able dicts."""

from cx_pangram.eval import (
    DatasetReport,
    RefsReport,
    Scored,
    SmokeReport,
    compute_dataset_metrics,
    format_dataset_report,
    format_refs_report,
    format_smoke_report,
    propose_bands,
)


def _rows():
    centers = {0: 0.05, 1: 0.30, 2: 0.60, 3: 0.92}
    cosines = {0: 0.01, 1: 0.06, 2: 0.12, 3: 0.30}
    rows = []
    for b in range(4):
        for _ in range(4):
            s = centers[b]
            rows.append(
                Scored(
                    preview="x",
                    n_words=100,
                    reliable=True,
                    cosine_score=cosines[b],
                    gt_bucket=b,
                    model_score=s,
                    model_bucket=round(s * 3),
                    model_bucket_argmax=round(s * 3),
                    probs=[],
                    latency_s=0.01,
                )
            )
    return rows


def _dataset_report():
    rows = _rows()
    return DatasetReport(
        n_total=len(rows),
        n_used=len(rows),
        reliable_only=True,
        split="test",
        seed=0,
        model="llama",
        device="cpu",
        quantize=None,
        quantized=False,
        metrics=compute_dataset_metrics(rows),
        bands=propose_bands(rows),
    )


def test_format_dataset_report_mentions_core_metrics():
    out = format_dataset_report(_dataset_report())
    assert "dataset eval" in out
    assert "bucket accuracy" in out
    assert "argmax variant" in out
    assert "auroc" in out
    assert "calibration" in out
    assert "confusion" in out
    assert "proposed bands" in out
    assert "interpolated" in out
    assert "pinned" in out


def test_format_dataset_report_renders_latency_and_cis():
    report = _dataset_report()
    report.latency = {"p50_s": 0.05, "p95_s": 0.09, "words_per_s": 1200.0}
    out = format_dataset_report(report)
    assert "latency" in out
    assert "words/s" in out
    assert "[0." in out


def test_dataset_report_to_dict_is_serializable():
    import json

    d = _dataset_report().to_dict()
    assert d["mode"] == "dataset"
    assert d["metrics"]["n"] == 16
    json.dumps(d)


def test_format_refs_report_and_dict():
    import json

    report = RefsReport(
        n_scored=2,
        n_skipped=1,
        reliable_only=False,
        model="roberta",
        device="cpu",
        overall={
            "n": 2,
            "min": 0.1,
            "q25": 0.1,
            "median": 0.2,
            "mean": 0.2,
            "q75": 0.3,
            "q90": 0.3,
            "max": 0.3,
        },
        fp_table=[(0.1, 1), (0.4, 0), (0.5, 0), (0.7, 0), (0.9, 0)],
        by_source={},
        results=[{"label": "a", "score": 0.1, "preview": "p"}],
        skipped=[{"label": "b", "reason": "no prose", "preview": None}],
    )
    out = format_refs_report(report)
    assert "refs eval" in out
    assert "overall model_score" in out
    assert "skipped" in out and "no prose" in out
    d = report.to_dict()
    assert d["mode"] == "refs"
    assert all("preview" not in r for r in d["results"])
    json.dumps(d)


def test_format_smoke_report_pass_fail():
    report = SmokeReport(
        labels=["human", "light", "vivid", "rewrite"],
        scores=[0.1, 0.3, 0.5, 0.8],
        bands=["human", "lightly edited", "moderately edited", "heavily edited"],
        human_is_min=True,
        rewrite_is_max=True,
        spearman_order=1.0,
        passed=True,
        model="llama",
        device="cpu",
    )
    out = format_smoke_report(report)
    assert "smoke gradient" in out
    assert "PASS" in out
    assert "rewrite" in out

    failed = SmokeReport(
        labels=report.labels,
        scores=[0.5, 0.3, 0.5, 0.4],
        bands=report.bands,
        human_is_min=False,
        rewrite_is_max=False,
        spearman_order=-0.2,
        passed=False,
        model="roberta",
        device="cpu",
    )
    assert "FAIL" in format_smoke_report(failed)


def test_bands_artifact_schema():
    import json

    from cx_pangram.eval import bands_artifact

    art = bands_artifact(_dataset_report())
    assert art["kind"] == "cx-pangram-bands"
    assert art["version"] == 1
    assert len(art["cuts"]) == 4
    assert all({"lt", "band", "pinned"} <= set(c) for c in art["cuts"])
    assert art["provenance"]["dataset"] == "pangram/editlens_iclr"
    json.dumps(art)


def test_markdown_summary_is_a_table():
    from cx_pangram.eval import format_markdown_summary

    out = format_markdown_summary(_dataset_report())
    assert out.startswith("| model |")
    assert "Reproduce: `cx-pangram eval" in out
    assert "[/" not in out

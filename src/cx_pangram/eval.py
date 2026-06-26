"""Faithfulness eval for the local EditLens decode against Pangram's labeled set.

Two modes share this module:

- **labeled dataset** (``eval_dataset``): score ``pangram/editlens_iclr`` rows whose
  continuous ``cosine_score`` (Δ = 1 − cosine_sim) carries a ground-truth bucket, then
  report how well the local decode reproduces EditLens.
- **unlabeled refs** (``eval_refs``): in-domain distribution / false-positive read
  from :func:`cx_pangram.lens.score_refs`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

DATASET_ID = "pangram/editlens_iclr"
N_BUCKETS = 4

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
_SAMPLES_DIR = os.path.join(_REPO_ROOT, "samples")

_FP_THRESHOLDS = (0.10, 0.40, 0.50, 0.70, 0.90)
_SMOKE_FILES = (
    "human_ishiguro.txt",
    "ishiguro_edit_light.txt",
    "ishiguro_edit_vivid.txt",
    "ishiguro_edit_rewrite.txt",
)


def score_to_bucket(
    s: float, lo_th: float = 0.03, hi_th: float = 0.15, n_buckets: int = N_BUCKETS
) -> int:
    """Discretize a continuous EditLens ``cosine_score`` into a bucket index.

    At or below ``lo_th`` is fully human (0); at or above ``hi_th`` is the top AI
    bucket (``n_buckets-1``); the open interval ``(lo_th, hi_th)`` is split evenly
    across the interior buckets. Mirrors EditLens preprocess (``<=`` / ``>=`` edges).
    """
    if s <= lo_th:
        return 0
    if s >= hi_th:
        return n_buckets - 1
    frac = (s - lo_th) / (hi_th - lo_th)
    return 1 + int(frac * (n_buckets - 2))


def _ensure_hf_token() -> None:
    """Fail fast with guidance when no Hugging Face token is resolvable.

    Delegates to ``huggingface_hub.get_token()``, which checks ``HF_TOKEN`` and
    the ``hf auth login`` credential cache — the dataset is gated, so a token
    with granted access is required before any rows can stream."""
    from huggingface_hub import get_token

    if get_token() is None:
        raise RuntimeError(
            f"no HF token: the eval dataset ({DATASET_ID}) is gated; "
            "export HF_TOKEN or run `hf auth login`"
        )


@dataclass
class Sample:
    text: str
    cosine_score: float
    gt_bucket: int
    n_words: int
    source_text: str | None = None


@dataclass
class Scored:
    preview: str
    n_words: int
    reliable: bool
    cosine_score: float
    gt_bucket: int
    model_score: float
    model_bucket: int
    band: str
    source_score: float | None = None
    source_bucket: int | None = None


def load_sample(
    n: int, *, split: str = "test", seed: int = 0, buffer_size: int = 5000
) -> list[Sample]:
    """Stream ``n`` shuffled rows of the gated dataset into :class:`Sample`s.

    ``n_words`` is computed through the project's own ``preprocess`` so the
    reliability floor here matches what ``engine.detect`` will apply downstream.
    """
    _ensure_hf_token()
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the eval dataset path needs `datasets`; install cx-pangram[eval]"
        ) from exc

    from .preprocess import clean_text, count_words

    try:
        ds = load_dataset(DATASET_ID, split=split, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not open {DATASET_ID}:{split} (gated — accept the terms and "
            f"ensure HF_TOKEN has access): {exc}"
        ) from exc

    ds = ds.shuffle(seed=seed, buffer_size=buffer_size).take(n)
    samples: list[Sample] = []
    for row in ds:
        cosine = float(row["cosine_score"])
        n_words = count_words(clean_text(row["text"]))
        samples.append(
            Sample(
                text=row["text"],
                cosine_score=cosine,
                gt_bucket=score_to_bucket(cosine),
                n_words=n_words,
                source_text=row.get("source_text"),
            )
        )
    return samples


def _score_one(engine, sample: Sample, with_source: bool) -> Scored:
    det = engine.detect(sample.text)
    src_score = src_bucket = None
    if with_source and sample.source_text:
        sdet = engine.detect(sample.source_text)
        src_score, src_bucket = sdet.score, sdet.bucket
    return Scored(
        preview=_flat(sample.text)[:80],
        n_words=det.n_words,
        reliable=det.reliable,
        cosine_score=sample.cosine_score,
        gt_bucket=sample.gt_bucket,
        model_score=det.score,
        model_bucket=det.bucket,
        band=det.band,
        source_score=src_score,
        source_bucket=src_bucket,
    )


def score_samples(
    samples: list[Sample],
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    quantize: bool | None = None,
    with_source: bool = False,
    engine=None,
) -> list[Scored]:
    """Score every sample through one shared :class:`~cx_pangram.engine.EditLens`.

    Pass a prebuilt ``engine`` to reuse a load (and to retain ownership for an
    explicit free, as :func:`compare_quant` does between its two runs).
    """
    if engine is None:
        from .engine import EditLens

        engine = EditLens(model=model, base=base, device=device, quantize=quantize)
    if engine.n_buckets != N_BUCKETS:
        raise AssertionError(
            f"eval assumes {N_BUCKETS} buckets; engine reports {engine.n_buckets}"
        )
    return [_score_one(engine, s, with_source) for s in samples]


def _rankdata(a) -> np.ndarray:
    """Average ranks with tie handling; matches ``scipy.stats.rankdata``."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(x, y) -> float:
    """Spearman rank correlation: Pearson on average ranks (no scipy runtime dep)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("spearman inputs must be the same length")
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _confusion(gt, pred, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """``n_buckets``×``n_buckets`` count matrix; rows = ground truth, cols = model."""
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)
    m = np.zeros((n_buckets, n_buckets), dtype=int)
    for g, p in zip(gt, pred):
        if not (0 <= g < n_buckets and 0 <= p < n_buckets):
            raise ValueError(
                f"bucket index out of range for {n_buckets} buckets: gt={g}, pred={p}"
            )
        m[g, p] += 1
    return m


def _describe(xs) -> dict:
    """n / min / q25 / median / mean / q75 / q90 / max for a score distribution."""
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        keys = ("min", "q25", "median", "mean", "q75", "q90", "max")
        return {"n": 0, **{k: float("nan") for k in keys}}
    return {
        "n": int(xs.size),
        "min": float(xs.min()),
        "q25": float(np.percentile(xs, 25)),
        "median": float(np.median(xs)),
        "mean": float(xs.mean()),
        "q75": float(np.percentile(xs, 75)),
        "q90": float(np.percentile(xs, 90)),
        "max": float(xs.max()),
    }


def _fp_table(
    xs, thresholds: tuple[float, ...] = _FP_THRESHOLDS
) -> list[tuple[float, int]]:
    """Count of scores at or above each threshold; on all-human input these are
    false positives. Ported from the are.na distribution probe."""
    xs = np.asarray(xs, dtype=float)
    return [(t, int((xs >= t).sum())) for t in thresholds]


def _balacc_sweep(
    neg, pos, lo: float = 0.0, hi: float = 1.0, step: float = 0.01
) -> dict | None:
    """Threshold over ``[lo, hi]`` maximizing balanced accuracy (neg below, pos at/above).

    Returns ``None`` when either class is empty. Generalized from the are.na
    human-vs-AI separation sweep.
    """
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return None
    best: dict | None = None
    for t in np.round(np.arange(lo, hi + 1e-9, step), 4):
        tp = int((pos >= t).sum())
        tn = int((neg < t).sum())
        tpr = tp / pos.size
        tnr = tn / neg.size
        bal = (tpr + tnr) / 2.0
        if best is None or bal > best["bal_acc"]:
            best = {
                "threshold": float(t),
                "bal_acc": float(bal),
                "tp": tp,
                "n_pos": int(pos.size),
                "fp": int(neg.size - tn),
                "n_neg": int(neg.size),
            }
    return best


@dataclass
class DatasetMetrics:
    n: int
    bucket_accuracy: float
    adjacent_accuracy: float
    bucket_mae: float
    confusion: list[list[int]]
    spearman_score: float
    spearman_bucket: float
    per_gt_score: dict[int, dict]
    source_control: dict | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "bucket_accuracy": self.bucket_accuracy,
            "adjacent_accuracy": self.adjacent_accuracy,
            "bucket_mae": self.bucket_mae,
            "confusion": self.confusion,
            "spearman_score": self.spearman_score,
            "spearman_bucket": self.spearman_bucket,
            "per_gt_score": {str(k): v for k, v in self.per_gt_score.items()},
            "source_control": self.source_control,
        }


def compute_dataset_metrics(
    scored: list[Scored], n_buckets: int = N_BUCKETS, with_source: bool = False
) -> DatasetMetrics:
    """Agreement between the local decode and the dataset ground truth.

    Buckets are compared directly (same ``0..n_buckets-1`` index space); the
    continuous ``cosine_score`` is compared to ``model_score`` only through Spearman
    rank correlation, never by subtraction across the two scales. ``model_bucket`` is
    the engine's rounded expected bucket (``round(score·(n-1))``), not ``argmax``.
    """
    if not scored:
        return DatasetMetrics(
            n=0,
            bucket_accuracy=float("nan"),
            adjacent_accuracy=float("nan"),
            bucket_mae=float("nan"),
            confusion=[[0] * n_buckets for _ in range(n_buckets)],
            spearman_score=float("nan"),
            spearman_bucket=float("nan"),
            per_gt_score={},
            source_control=None,
        )
    gt = np.array([s.gt_bucket for s in scored])
    mb = np.array([s.model_bucket for s in scored])
    ms = np.array([s.model_score for s in scored], dtype=float)
    cs = np.array([s.cosine_score for s in scored], dtype=float)

    per_gt: dict[int, dict] = {}
    for b in range(n_buckets):
        xs = ms[gt == b]
        if xs.size:
            per_gt[b] = _describe(xs)

    source_control = None
    if with_source:
        src_scores = [s.source_score for s in scored if s.source_score is not None]
        if src_scores:
            src_buckets = [
                s.source_bucket for s in scored if s.source_bucket is not None
            ]
            source_control = {
                "describe": _describe(src_scores),
                "frac_bucket0": float((np.asarray(src_buckets) == 0).mean()),
            }

    return DatasetMetrics(
        n=len(scored),
        bucket_accuracy=float((gt == mb).mean()),
        adjacent_accuracy=float((np.abs(gt - mb) <= 1).mean()),
        bucket_mae=float(np.abs(gt - mb).mean()),
        confusion=_confusion(gt, mb, n_buckets).tolist(),
        spearman_score=_spearman(ms, cs),
        spearman_bucket=_spearman(mb.astype(float), gt.astype(float)),
        per_gt_score=per_gt,
        source_control=source_control,
    )


@dataclass
class BandProposal:
    boundaries: list[tuple[float, str, bool]]
    sweep_human: dict | None
    sweep_ai: dict | None

    def to_dict(self) -> dict:
        return {
            "boundaries": [
                {"cut": c, "band_below": name, "pinned": pinned}
                for c, name, pinned in self.boundaries
            ],
            "sweep_human": self.sweep_human,
            "sweep_ai": self.sweep_ai,
        }


def propose_bands(scored: list[Scored], n_buckets: int = N_BUCKETS) -> BandProposal:
    """Derive band cuts on the model-score scale from the ground-truth buckets.

    With only ``n_buckets`` ground-truth levels, the human cut (bucket 0 vs the rest)
    and the top-AI cut (bucket ``n_buckets-1`` vs the rest) are empirically pinned by a
    balanced-accuracy sweep; the interior product cuts are linearly interpolated between
    them and flagged ``pinned=False`` so the report can label them as such.
    """
    from .engine import _BANDS

    scores = np.array([s.model_score for s in scored], dtype=float)
    gts = np.array([s.gt_bucket for s in scored], dtype=int)

    sweep_human = _balacc_sweep(scores[gts == 0], scores[gts >= 1])
    sweep_ai = _balacc_sweep(scores[gts <= n_buckets - 2], scores[gts == n_buckets - 1])

    t01 = sweep_human["threshold"] if sweep_human else _BANDS[0][0]
    t_top = sweep_ai["threshold"] if sweep_ai else _BANDS[-1][0]
    t_top = max(t_top, t01)

    names = [name for _, name in _BANDS]
    span = t_top - t01
    n_cuts = len(names)
    boundaries: list[tuple[float, str, bool]] = []
    for i, name in enumerate(names):
        if i == 0:
            cut, pinned = t01, sweep_human is not None
        elif i == n_cuts - 1:
            cut, pinned = t_top, sweep_ai is not None
        else:
            cut, pinned = t01 + span * (i / (n_cuts - 1)), False
        boundaries.append((round(cut, 3), name, pinned))
    return BandProposal(
        boundaries=boundaries, sweep_human=sweep_human, sweep_ai=sweep_ai
    )


@dataclass
class QuantComparison:
    model: str
    device: str
    n: int
    score_mae: float
    max_delta: float
    bucket_flip_rate: float
    metrics_quant: DatasetMetrics
    metrics_bf16: DatasetMetrics
    scored_quant: list[Scored] = field(default_factory=list, repr=False)
    scored_bf16: list[Scored] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "n": self.n,
            "score_mae": self.score_mae,
            "max_delta": self.max_delta,
            "bucket_flip_rate": self.bucket_flip_rate,
            "metrics_quant": self.metrics_quant.to_dict(),
            "metrics_bf16": self.metrics_bf16.to_dict(),
        }


def compare_quant(
    samples: list[Sample],
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    with_source: bool = False,
) -> QuantComparison:
    """Score the same samples 4-bit then bf16 on one CUDA device and diff them.

    Frees the quantized engine before loading the unquantized one so both fit. Raises
    off CUDA or on the roberta backbone, neither of which has a 4-bit path to compare.
    """
    import gc

    import torch

    from .engine import EditLens, select_device

    resolved = select_device(device)
    if not resolved.startswith("cuda") or torch.version.hip is not None:
        raise RuntimeError(
            "compare_quant diffs 4-bit against bf16 and needs a real CUDA device"
        )
    if (model or "") == "roberta":
        raise RuntimeError(
            "compare_quant targets the llama backbone; roberta has no 4-bit path"
        )

    eng_q = EditLens(model=model, base=base, device=device, quantize=True)
    used_model, used_device = eng_q.model_name, eng_q.device
    scored_q = score_samples(samples, engine=eng_q, with_source=with_source)
    del eng_q
    gc.collect()
    torch.cuda.empty_cache()

    eng_f = EditLens(model=model, base=base, device=device, quantize=False)
    scored_f = score_samples(samples, engine=eng_f, with_source=with_source)
    del eng_f
    gc.collect()
    torch.cuda.empty_cache()

    q_scores = np.array([s.model_score for s in scored_q], dtype=float)
    f_scores = np.array([s.model_score for s in scored_f], dtype=float)
    deltas = np.abs(q_scores - f_scores)
    flips = sum(q.model_bucket != f.model_bucket for q, f in zip(scored_q, scored_f))
    n = len(samples)
    return QuantComparison(
        model=used_model,
        device=used_device,
        n=n,
        score_mae=float(deltas.mean()) if n else float("nan"),
        max_delta=float(deltas.max()) if n else float("nan"),
        bucket_flip_rate=flips / n if n else float("nan"),
        metrics_quant=compute_dataset_metrics(scored_q, with_source=with_source),
        metrics_bf16=compute_dataset_metrics(scored_f, with_source=with_source),
        scored_quant=scored_q,
        scored_bf16=scored_f,
    )


@dataclass
class DatasetReport:
    n_total: int
    n_used: int
    reliable_only: bool
    split: str
    seed: int
    model: str
    device: str
    quantize: bool | None
    quantized: bool
    metrics: DatasetMetrics
    bands: BandProposal
    quant_compare: QuantComparison | None = None

    def to_dict(self) -> dict:
        return {
            "mode": "dataset",
            "n_total": self.n_total,
            "n_used": self.n_used,
            "reliable_only": self.reliable_only,
            "split": self.split,
            "seed": self.seed,
            "model": self.model,
            "device": self.device,
            "quantize": self.quantize,
            "quantized": self.quantized,
            "metrics": self.metrics.to_dict(),
            "bands": self.bands.to_dict(),
            "quant_compare": self.quant_compare.to_dict()
            if self.quant_compare
            else None,
        }


def eval_dataset(
    *,
    n: int,
    split: str = "test",
    seed: int = 0,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    quantize: bool | None = None,
    reliable_only: bool = True,
    compare: bool = False,
    with_source: bool = False,
) -> DatasetReport:
    """End-to-end labeled-dataset eval: load, score, measure, propose bands."""
    samples = load_sample(n, split=split, seed=seed)
    quant_compare = None
    if compare:
        quant_compare = compare_quant(
            samples, model=model, base=base, device=device, with_source=with_source
        )
        scored = quant_compare.scored_quant
        used_model, used_device = quant_compare.model, quant_compare.device
        used_quantized = True
    else:
        from .engine import EditLens

        engine = EditLens(model=model, base=base, device=device, quantize=quantize)
        used_model, used_device = engine.model_name, engine.device
        used_quantized = engine.quantized
        scored = score_samples(samples, engine=engine, with_source=with_source)

    used = [s for s in scored if s.reliable] if reliable_only else scored
    metrics = compute_dataset_metrics(used, with_source=with_source)
    bands = propose_bands(used)
    return DatasetReport(
        n_total=len(scored),
        n_used=len(used),
        reliable_only=reliable_only,
        split=split,
        seed=seed,
        model=used_model,
        device=used_device,
        quantize=quantize,
        quantized=used_quantized,
        metrics=metrics,
        bands=bands,
        quant_compare=quant_compare,
    )


@dataclass
class RefsReport:
    n_scored: int
    n_skipped: int
    reliable_only: bool
    model: str | None
    device: str | None
    overall: dict
    fp_table: list[tuple[float, int]]
    by_source: dict[str, dict]
    results: list[dict]
    skipped: list[dict]

    def to_dict(self) -> dict:
        return {
            "mode": "refs",
            "n_scored": self.n_scored,
            "n_skipped": self.n_skipped,
            "reliable_only": self.reliable_only,
            "model": self.model,
            "device": self.device,
            "overall": self.overall,
            "fp_table": [{"threshold": t, "count": c} for t, c in self.fp_table],
            "by_source": self.by_source,
            "results": [
                {k: v for k, v in r.items() if k != "preview"} for r in self.results
            ],
            "skipped": [
                {k: v for k, v in r.items() if k != "preview"} for r in self.skipped
            ],
        }


def eval_refs(
    targets,
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    quantize: bool | None = None,
    split: bool = False,
    reliable_only: bool = False,
) -> RefsReport:
    """Unlabeled in-domain distribution / false-positive read over resolved refs."""
    from . import lens as lens_core

    results = lens_core.score_refs(
        targets,
        model=model,
        base=base,
        device=device,
        split=split,
        quantize=quantize,
    )
    scored = [r for r in results if not r["skipped"] and r["score"] is not None]
    skipped = [r for r in results if r["skipped"]]
    used = [r for r in scored if r["reliable"]] if reliable_only else scored

    by_source: dict[str, list[float]] = {}
    for r in used:
        key = r.get("source") or r.get("label") or "<ref>"
        by_source.setdefault(key, []).append(r["score"])

    model_used = next((r.get("model") for r in scored if r.get("model")), model)
    return RefsReport(
        n_scored=len(scored),
        n_skipped=len(skipped),
        reliable_only=reliable_only,
        model=model_used,
        device=device,
        overall=_describe([r["score"] for r in used]),
        fp_table=_fp_table([r["score"] for r in used]),
        by_source={k: _describe(v) for k, v in by_source.items()},
        results=used,
        skipped=skipped,
    )


@dataclass
class SmokeReport:
    labels: list[str]
    scores: list[float]
    bands: list[str]
    human_is_min: bool
    rewrite_is_max: bool
    spearman_order: float
    passed: bool
    model: str
    device: str

    def to_dict(self) -> dict:
        return {
            "mode": "smoke",
            "labels": self.labels,
            "scores": self.scores,
            "bands": self.bands,
            "human_is_min": self.human_is_min,
            "rewrite_is_max": self.rewrite_is_max,
            "spearman_order": self.spearman_order,
            "passed": self.passed,
            "model": self.model,
            "device": self.device,
        }


def smoke_gradient(
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    quantize: bool | None = None,
    samples_dir: str | None = None,
) -> SmokeReport:
    """Score the human Ishiguro paragraph and its three AI edits as a monotonicity probe.

    Absolute values differ from the unreleased 24B; the pass condition is structural:
    the human source reads lowest and the full rewrite reads highest.
    """
    from .engine import EditLens

    directory = samples_dir or _SAMPLES_DIR
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"smoke samples not found at {directory}; run from the repo checkout"
        )
    texts: list[str] = []
    for name in _SMOKE_FILES:
        with open(os.path.join(directory, name)) as fh:
            texts.append(fh.read())

    engine = EditLens(model=model, base=base, device=device, quantize=quantize)
    dets = [engine.detect(t) for t in texts]
    scores = [d.score for d in dets]
    bands = [d.band for d in dets]
    human_is_min = scores[0] == min(scores)
    rewrite_is_max = scores[-1] == max(scores)
    return SmokeReport(
        labels=list(_SMOKE_FILES),
        scores=scores,
        bands=bands,
        human_is_min=human_is_min,
        rewrite_is_max=rewrite_is_max,
        spearman_order=_spearman(scores, list(range(len(scores)))),
        passed=human_is_min and rewrite_is_max,
        model=engine.model_name,
        device=engine.device,
    )


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fmt(x: float, places: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{places}f}"


def _describe_line(d: dict) -> str:
    return (
        f"n={d['n']:<5} min={_fmt(d['min'])} q25={_fmt(d['q25'])} "
        f"median={_fmt(d['median'])} mean={_fmt(d['mean'])} "
        f"q75={_fmt(d['q75'])} q90={_fmt(d['q90'])} max={_fmt(d['max'])}"
    )


def format_dataset_report(report: DatasetReport) -> str:
    """Human-readable dataset eval summary with rich markup."""
    m = report.metrics
    lines = [
        f"[bold]dataset eval[/bold]  {report.model} · {report.device} · "
        f"{'4-bit' if report.quantized else 'unquantized'} · "
        f"split={report.split} seed={report.seed}",
        f"scored {report.n_used}/{report.n_total} "
        f"({'reliable only' if report.reliable_only else 'all lengths'})",
        "",
        f"bucket accuracy   {_fmt(m.bucket_accuracy)}",
        f"adjacent (±1)     {_fmt(m.adjacent_accuracy)}",
        f"bucket MAE        {_fmt(m.bucket_mae)}",
        f"spearman score    {_fmt(m.spearman_score)}   [dim](model_score vs cosine_score)[/dim]",
        f"spearman bucket   {_fmt(m.spearman_bucket)}",
        "",
        "[bold]confusion[/bold] [dim](rows=ground truth, cols=model)[/dim]",
    ]
    for i, row in enumerate(m.confusion):
        lines.append(f"  gt{i}  " + " ".join(f"{c:>5}" for c in row))
    lines.append("")
    lines.append("[bold]model_score within each ground-truth bucket[/bold]")
    for b in sorted(m.per_gt_score):
        lines.append(f"  gt{b}  {_describe_line(m.per_gt_score[b])}")
    if m.source_control:
        lines.append("")
        lines.append(
            f"[bold]source_text control[/bold] (should read human): "
            f"frac in bucket0 = {_fmt(m.source_control['frac_bucket0'])}"
        )
        lines.append("  " + _describe_line(m.source_control["describe"]))
    lines.append("")
    lines.append("[bold]proposed bands[/bold] (model-score scale)")
    for cut, name, pinned in report.bands.boundaries:
        tag = "pinned" if pinned else "interpolated"
        lines.append(f"  < {_fmt(cut)}  {name:<18} [dim]({tag})[/dim]")
    if report.quant_compare:
        qc = report.quant_compare
        lines.append("")
        lines.append(
            f"[bold]4-bit vs bf16[/bold]  score MAE={_fmt(qc.score_mae)} "
            f"max Δ={_fmt(qc.max_delta)} bucket-flip={_fmt(qc.bucket_flip_rate)}"
        )
    return "\n".join(lines)


def format_refs_report(report: RefsReport) -> str:
    """Human-readable refs distribution / false-positive summary with rich markup."""
    lines = [
        f"[bold]refs eval[/bold]  {report.model or 'auto'} · "
        f"{report.n_scored} scored, {report.n_skipped} skipped "
        f"({'reliable only' if report.reliable_only else 'all lengths'})",
        "",
        "[bold]overall model_score[/bold]",
        "  " + _describe_line(report.overall),
        "",
        "[bold]elevation counts[/bold] [dim](score ≥ threshold)[/dim]",
    ]
    for t, c in report.fp_table:
        lines.append(f"  ≥ {_fmt(t, 2)}  {c}")
    if len(report.by_source) > 1:
        lines.append("")
        lines.append("[bold]by source[/bold]")
        for key, desc in report.by_source.items():
            lines.append(f"  {key}")
            lines.append("    " + _describe_line(desc))
    if report.skipped:
        lines.append("")
        lines.append("[bold]skipped[/bold]")
        for r in report.skipped:
            lines.append(f"  [dim]{r.get('label')} — {r.get('reason')}[/dim]")
    return "\n".join(lines)


def format_smoke_report(report: SmokeReport) -> str:
    """Human-readable monotonicity-probe summary with rich markup."""
    verdict = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    lines = [
        f"[bold]smoke gradient[/bold]  {report.model} · {report.device}   {verdict}",
        "",
    ]
    for label, score, band in zip(report.labels, report.scores, report.bands):
        lines.append(f"  {_fmt(score)}  {band:<18} {label}")
    lines.append("")
    lines.append(
        f"human lowest = {report.human_is_min} · rewrite highest = "
        f"{report.rewrite_is_max} · spearman-with-order = {_fmt(report.spearman_order)}"
    )
    return "\n".join(lines)

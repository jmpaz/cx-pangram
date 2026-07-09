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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import EditLens

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
    model_bucket_argmax: int
    probs: list[float]
    latency_s: float
    source_score: float | None = None
    source_bucket: int | None = None


def load_samples(
    n: int, *, split: str = "test", seed: int = 0, buffer_size: int = 5000
) -> list[Sample]:
    """Stream ``n`` shuffled rows of the gated dataset into :class:`Sample`s.

    ``n_words`` is computed through the project's own ``preprocess`` so the
    reliability floor here matches what ``engine.detect`` will apply downstream.

    ``shuffle(buffer_size=...)`` is a streaming approximation: rows are drawn
    from a rolling buffer, not uniformly from the whole split. Deterministic
    under ``seed``, but not an unbiased sample when the split is much larger
    than the buffer.
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


def _weighted_probs(det) -> list[float]:
    """Length-weighted mean of per-chunk softmax vectors; the document-level
    bucket distribution that ``argmax`` metrics and ECE read from."""
    if not det.chunks:
        return []
    weights = np.array([c.n_words for c in det.chunks], dtype=float)
    probs = np.array([c.probs for c in det.chunks], dtype=float)
    total = weights.sum() or 1.0
    return (probs * weights[:, None]).sum(axis=0) / total


def _score_one(engine: EditLens, sample: Sample, with_source: bool) -> Scored:
    t0 = time.perf_counter()
    det = engine.detect(sample.text)
    latency_s = time.perf_counter() - t0
    src_score = src_bucket = None
    if with_source and sample.source_text:
        sdet = engine.detect(sample.source_text)
        src_score, src_bucket = sdet.score, sdet.bucket
    probs = _weighted_probs(det)
    return Scored(
        preview=_flat(sample.text)[:80],
        n_words=det.n_words,
        reliable=det.reliable,
        cosine_score=sample.cosine_score,
        gt_bucket=sample.gt_bucket,
        model_score=det.score,
        model_bucket=det.bucket,
        model_bucket_argmax=int(np.argmax(probs)) if len(probs) else det.bucket,
        probs=[float(p) for p in probs],
        latency_s=latency_s,
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
    for g, p in zip(gt, pred, strict=True):
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
    false positives."""
    xs = np.asarray(xs, dtype=float)
    return [(t, int((xs >= t).sum())) for t in thresholds]


def _auroc(neg, pos) -> float:
    """AUROC via the Mann–Whitney U statistic on average ranks (tie-correct).

    Probability that a random positive outscores a random negative; the
    threshold-free companion to the balanced-accuracy sweep."""
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([neg, pos]))
    r_pos = ranks[neg.size :].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _balacc_sweep(
    neg, pos, lo: float = 0.0, hi: float = 1.0, step: float = 0.01
) -> dict | None:
    """Threshold over ``[lo, hi]`` maximizing balanced accuracy (neg below, pos at/above).

    Ties across a plateau of equally good thresholds resolve to the plateau
    midpoint rather than its left edge, so the cut sits centrally in the score
    gap instead of hugging the negative class. Returns ``None`` when either
    class is empty.
    """
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return None
    ts = np.round(np.arange(lo, hi + 1e-9, step), 4)
    tpr = (pos[None, :] >= ts[:, None]).mean(axis=1)
    tnr = (neg[None, :] < ts[:, None]).mean(axis=1)
    bal = (tpr + tnr) / 2.0
    best_mask = bal == bal.max()
    plateau = ts[best_mask]
    t = float(np.round(np.median(plateau), 4))
    tp = int((pos >= t).sum())
    tn = int((neg < t).sum())
    return {
        "threshold": t,
        "bal_acc": float(bal.max()),
        "plateau": [float(plateau.min()), float(plateau.max())],
        "tp": tp,
        "n_pos": int(pos.size),
        "fp": int(neg.size - tn),
        "n_neg": int(neg.size),
    }


def _bootstrap_cis(
    stats: dict[str, Callable[[np.ndarray], float]],
    n: int,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Nonparametric bootstrap percentile CIs.

    Each stat receives an index array into the scored sample and returns a
    float; resampling is shared across stats so their CIs come from the same
    draws. NaN resamples (e.g. a draw with one class absent) are dropped
    per-stat before taking percentiles.
    """
    if n == 0:
        return {k: [float("nan"), float("nan")] for k in stats}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in stats}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for key, fn in stats.items():
            draws[key].append(fn(idx))
    out: dict[str, list[float]] = {}
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[key] = [float("nan"), float("nan")]
        else:
            lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[key] = [float(lo), float(hi)]
    return out


def _calibration_read(ms: np.ndarray, gt: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Quantile-binned reliability read: mean model score vs mean normalized
    ground-truth bucket per bin.

    ``model_score`` is an expected bucket index, not a probability, so classic
    calibration curves don't apply directly; this compares the two quantities
    on the shared ``[0, 1]`` scale within score quantiles. A well-calibrated
    decode tracks the diagonal.
    """
    if ms.size == 0:
        return []
    edges = np.percentile(ms, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    out: list[dict] = []
    gt_norm = gt / (N_BUCKETS - 1)
    for i in range(n_bins):
        mask = (ms >= edges[i]) & (ms < edges[i + 1])
        if not mask.any():
            continue
        out.append(
            {
                "n": int(mask.sum()),
                "mean_score": float(ms[mask].mean()),
                "mean_gt": float(gt_norm[mask].mean()),
            }
        )
    return out


def _ece(probs: np.ndarray, gt: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the argmax-bucket confidence.

    Uses the document-level weighted softmax: confidence = max prob, correct =
    argmax bucket equals ground truth. This is the one probability the decode
    actually emits, so it is the honest place to measure calibration.
    """
    if probs.size == 0:
        return float("nan")
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == gt
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


@dataclass
class DatasetMetrics:
    n: int
    bucket_accuracy: float
    bucket_accuracy_argmax: float
    adjacent_accuracy: float
    bucket_mae: float
    confusion: list[list[int]]
    spearman_score: float
    spearman_bucket: float
    auroc_human: float
    auroc_top: float
    ece_argmax: float
    calibration: list[dict]
    cis: dict[str, list[float]]
    per_gt_score: dict[int, dict]
    worst_disagreements: list[dict]
    source_control: dict | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "bucket_accuracy": self.bucket_accuracy,
            "bucket_accuracy_argmax": self.bucket_accuracy_argmax,
            "adjacent_accuracy": self.adjacent_accuracy,
            "bucket_mae": self.bucket_mae,
            "confusion": self.confusion,
            "spearman_score": self.spearman_score,
            "spearman_bucket": self.spearman_bucket,
            "auroc_human": self.auroc_human,
            "auroc_top": self.auroc_top,
            "ece_argmax": self.ece_argmax,
            "calibration": self.calibration,
            "cis": self.cis,
            "per_gt_score": {str(k): v for k, v in self.per_gt_score.items()},
            "worst_disagreements": self.worst_disagreements,
            "source_control": self.source_control,
        }


def _empty_metrics(n_buckets: int) -> DatasetMetrics:
    nan = float("nan")
    return DatasetMetrics(
        n=0,
        bucket_accuracy=nan,
        bucket_accuracy_argmax=nan,
        adjacent_accuracy=nan,
        bucket_mae=nan,
        confusion=[[0] * n_buckets for _ in range(n_buckets)],
        spearman_score=nan,
        spearman_bucket=nan,
        auroc_human=nan,
        auroc_top=nan,
        ece_argmax=nan,
        calibration=[],
        cis={},
        per_gt_score={},
        worst_disagreements=[],
        source_control=None,
    )


def compute_dataset_metrics(
    scored: list[Scored],
    n_buckets: int = N_BUCKETS,
    with_source: bool = False,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> DatasetMetrics:
    """Agreement between the local decode and the dataset ground truth.

    Buckets are compared directly (same ``0..n_buckets-1`` index space); the
    continuous ``cosine_score`` is compared to ``model_score`` only through Spearman
    rank correlation, never by subtraction across the two scales. Two bucket
    accuracies are reported: ``model_bucket`` (the engine's rounded expected bucket,
    ``round(score·(n-1))``) and ``model_bucket_argmax`` (mode of the weighted softmax,
    comparable to EditLens' own argmax-based confusion). Headline metrics carry
    seeded bootstrap percentile CIs.
    """
    if not scored:
        return _empty_metrics(n_buckets)
    gt = np.array([s.gt_bucket for s in scored])
    mb = np.array([s.model_bucket for s in scored])
    mba = np.array([s.model_bucket_argmax for s in scored])
    ms = np.array([s.model_score for s in scored], dtype=float)
    cs = np.array([s.cosine_score for s in scored], dtype=float)
    probs = np.array([s.probs for s in scored], dtype=float)

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

    def auroc_human(idx: np.ndarray) -> float:
        g, m = gt[idx], ms[idx]
        return _auroc(m[g == 0], m[g >= 1])

    def auroc_top(idx: np.ndarray) -> float:
        g, m = gt[idx], ms[idx]
        return _auroc(m[g <= n_buckets - 2], m[g == n_buckets - 1])

    cis = _bootstrap_cis(
        {
            "bucket_accuracy": lambda idx: float((gt[idx] == mb[idx]).mean()),
            "adjacent_accuracy": lambda idx: float(
                (np.abs(gt[idx] - mb[idx]) <= 1).mean()
            ),
            "spearman_score": lambda idx: _spearman(ms[idx], cs[idx]),
            "auroc_human": auroc_human,
            "auroc_top": auroc_top,
        },
        len(scored),
        n_boot=n_boot,
        seed=seed,
    )

    err = np.abs(gt - mb)
    worst_idx = np.argsort(-(err + np.abs(ms - gt / (n_buckets - 1))))[:3]
    worst = [
        {
            "preview": scored[i].preview,
            "gt_bucket": int(gt[i]),
            "model_bucket": int(mb[i]),
            "model_score": float(ms[i]),
            "cosine_score": float(cs[i]),
        }
        for i in worst_idx
        if err[i] > 0
    ]

    all_idx = np.arange(len(scored))
    return DatasetMetrics(
        n=len(scored),
        bucket_accuracy=float((gt == mb).mean()),
        bucket_accuracy_argmax=float((gt == mba).mean()),
        adjacent_accuracy=float((np.abs(gt - mb) <= 1).mean()),
        bucket_mae=float(np.abs(gt - mb).mean()),
        confusion=_confusion(gt, mb, n_buckets).tolist(),
        spearman_score=_spearman(ms, cs),
        spearman_bucket=_spearman(mb.astype(float), gt.astype(float)),
        auroc_human=auroc_human(all_idx),
        auroc_top=auroc_top(all_idx),
        ece_argmax=_ece(probs, gt) if probs.ndim == 2 else float("nan"),
        calibration=_calibration_read(ms, gt),
        cis=cis,
        per_gt_score=per_gt,
        worst_disagreements=worst,
        source_control=source_control,
    )


@dataclass
class BandProposal:
    boundaries: list[tuple[float, str, bool]]
    sweeps: dict[str, dict | None]
    cut_cis: dict[str, list[float]]
    degenerate: bool

    def to_dict(self) -> dict:
        return {
            "boundaries": [
                {"cut": c, "band_below": name, "pinned": pinned}
                for c, name, pinned in self.boundaries
            ],
            "sweeps": self.sweeps,
            "cut_cis": self.cut_cis,
            "degenerate": self.degenerate,
        }


_SWEEP_KEYS = ("human", "light", "top")


def _band_sweeps(
    scores: np.ndarray, gts: np.ndarray, n_buckets: int
) -> dict[str, dict | None]:
    """The three one-vs-rest cuts the four ordinal ground-truth buckets support:
    gt≤0|≥1 (human), gt≤1|≥2 (light), gt≤2|≥3 (top)."""
    return {
        "human": _balacc_sweep(scores[gts == 0], scores[gts >= 1]),
        "light": _balacc_sweep(scores[gts <= 1], scores[gts >= 2]),
        "top": _balacc_sweep(
            scores[gts <= n_buckets - 2], scores[gts == n_buckets - 1]
        ),
    }


def propose_bands(
    scored: list[Scored],
    n_buckets: int = N_BUCKETS,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> BandProposal:
    """Derive band cuts on the model-score scale from the ground-truth buckets.

    Four ordinal ground-truth buckets support exactly three one-vs-rest cuts, so
    three of the four band boundaries are empirically pinned by balanced-accuracy
    sweeps (plateau midpoints); only the ``moderately edited`` cut lacks ground
    truth and is interpolated between its pinned neighbors, flagged
    ``pinned=False``. Cuts are forced monotone non-decreasing; ``degenerate``
    reports when that clamp collapsed a band's span to zero. Each pinned cut
    carries a seeded bootstrap CI from re-running its sweep over resamples.
    """
    from .engine import BANDS

    scores = np.array([s.model_score for s in scored], dtype=float)
    gts = np.array([s.gt_bucket for s in scored], dtype=int)

    sweeps = _band_sweeps(scores, gts, n_buckets)
    defaults = {"human": BANDS[0][0], "light": BANDS[1][0], "top": BANDS[3][0]}
    cuts = {
        k: (sw["threshold"] if (sw := sweeps[k]) else defaults[k])
        for k in _SWEEP_KEYS
    }

    t_h, t_l, t_top = cuts["human"], cuts["light"], cuts["top"]
    t_l = max(t_l, t_h)
    t_top_c = max(t_top, t_l)
    degenerate = (t_l != cuts["light"]) or (t_top_c != cuts["top"]) or t_h == t_top_c
    t_top = t_top_c
    t_m = (t_l + t_top) / 2.0

    names = [name for _, name in BANDS]
    ordered = [
        (t_h, names[0], sweeps["human"] is not None),
        (t_l, names[1], sweeps["light"] is not None),
        (t_m, names[2], False),
        (t_top, names[3], sweeps["top"] is not None),
    ]
    boundaries = [(round(c, 3), name, pinned) for c, name, pinned in ordered]

    def cut_stat(key: str) -> Callable[[np.ndarray], float]:
        def stat(idx: np.ndarray) -> float:
            sw = _band_sweeps(scores[idx], gts[idx], n_buckets)[key]
            return sw["threshold"] if sw else float("nan")

        return stat

    cut_cis = _bootstrap_cis(
        {k: cut_stat(k) for k in _SWEEP_KEYS if sweeps[k] is not None},
        len(scored),
        n_boot=n_boot,
        seed=seed,
    )
    return BandProposal(
        boundaries=boundaries, sweeps=sweeps, cut_cis=cut_cis, degenerate=degenerate
    )


@dataclass
class RunComparison:
    """Paired A/B read over identical samples: two engine configs, one diff."""

    label_a: str
    label_b: str
    model: str
    device: str
    n: int
    score_mae: float
    max_delta: float
    bucket_flip_rate: float
    metrics_a: DatasetMetrics
    metrics_b: DatasetMetrics
    latency_a: dict = field(default_factory=dict)
    latency_b: dict = field(default_factory=dict)
    scored_a: list[Scored] = field(default_factory=list, repr=False)
    scored_b: list[Scored] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "model": self.model,
            "device": self.device,
            "n": self.n,
            "score_mae": self.score_mae,
            "max_delta": self.max_delta,
            "bucket_flip_rate": self.bucket_flip_rate,
            "metrics_a": self.metrics_a.to_dict(),
            "metrics_b": self.metrics_b.to_dict(),
            "latency_a": self.latency_a,
            "latency_b": self.latency_b,
        }


def _compare_runs(
    samples: list[Sample],
    kwargs_a: dict,
    kwargs_b: dict,
    labels: tuple[str, str],
    *,
    with_source: bool = False,
    reliable_only: bool = True,
) -> RunComparison:
    """Score identical samples through two engine configs and diff them, freeing
    engine A before loading engine B so both fit on one device.

    ``reliable_only`` applies the same population filter as the headline metrics
    so every metric block in one report describes the same rows. Score MAE and
    bucket-flip rate are paired (same row, run A vs run B); the per-run metric
    blocks and latency summaries are per-config.
    """
    import gc

    import torch

    from .engine import EditLens

    def free() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eng_a = EditLens(**kwargs_a)
    used_model, used_device = eng_a.model_name, eng_a.device
    scored_a = score_samples(samples, engine=eng_a, with_source=with_source)
    del eng_a
    free()

    eng_b = EditLens(**kwargs_b)
    scored_b = score_samples(samples, engine=eng_b, with_source=with_source)
    del eng_b
    free()

    if reliable_only:
        keep = [
            i
            for i in range(len(samples))
            if scored_a[i].reliable and scored_b[i].reliable
        ]
        scored_a = [scored_a[i] for i in keep]
        scored_b = [scored_b[i] for i in keep]

    a_scores = np.array([s.model_score for s in scored_a], dtype=float)
    b_scores = np.array([s.model_score for s in scored_b], dtype=float)
    deltas = np.abs(a_scores - b_scores)
    flips = sum(
        a.model_bucket != b.model_bucket
        for a, b in zip(scored_a, scored_b, strict=True)
    )
    n = len(scored_a)
    return RunComparison(
        label_a=labels[0],
        label_b=labels[1],
        model=used_model,
        device=used_device,
        n=n,
        score_mae=float(deltas.mean()) if n else float("nan"),
        max_delta=float(deltas.max()) if n else float("nan"),
        bucket_flip_rate=flips / n if n else float("nan"),
        metrics_a=compute_dataset_metrics(scored_a, with_source=with_source),
        metrics_b=compute_dataset_metrics(scored_b, with_source=with_source),
        latency_a=_latency_summary(scored_a),
        latency_b=_latency_summary(scored_b),
        scored_a=scored_a,
        scored_b=scored_b,
    )


def compare_quant(
    samples: list[Sample],
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    with_source: bool = False,
    reliable_only: bool = True,
) -> RunComparison:
    """Diff 4-bit against bf16 on the same samples and one CUDA device.

    Raises off CUDA or on the roberta backbone, neither of which has a 4-bit
    path to compare.
    """
    import torch

    from .engine import select_device

    resolved = select_device(device)
    if not resolved.startswith("cuda") or torch.version.hip is not None:
        raise RuntimeError(
            "compare_quant diffs 4-bit against bf16 and needs a real CUDA device"
        )
    if (model or "") == "roberta":
        raise RuntimeError(
            "compare_quant targets the llama backbone; roberta has no 4-bit path"
        )
    common = {"model": model, "base": base, "device": device}
    return _compare_runs(
        samples,
        {**common, "quantize": True},
        {**common, "quantize": False},
        ("4-bit", "bf16"),
        with_source=with_source,
        reliable_only=reliable_only,
    )


def compare_models(
    samples: list[Sample],
    *,
    device: str | None = None,
    quantize: bool | None = None,
    with_source: bool = False,
    reliable_only: bool = True,
) -> RunComparison:
    """llama vs roberta head-to-head on identical samples: paired score deltas,
    per-model agreement metrics, and per-model latency."""
    return _compare_runs(
        samples,
        {"model": "llama", "device": device, "quantize": quantize},
        {"model": "roberta", "device": device},
        ("llama", "roberta"),
        with_source=with_source,
        reliable_only=reliable_only,
    )


def _latency_summary(scored: list[Scored]) -> dict:
    """p50/p95 per-doc latency and aggregate throughput in words/sec."""
    if not scored:
        return {
            "p50_s": float("nan"),
            "p95_s": float("nan"),
            "words_per_s": float("nan"),
        }
    lat = np.array([s.latency_s for s in scored], dtype=float)
    words = sum(s.n_words for s in scored)
    total = lat.sum()
    return {
        "p50_s": float(np.percentile(lat, 50)),
        "p95_s": float(np.percentile(lat, 95)),
        "words_per_s": float(words / total) if total else float("nan"),
    }


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
    latency: dict = field(default_factory=dict)
    compare: RunComparison | None = None

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
            "latency": self.latency,
            "compare": self.compare.to_dict() if self.compare else None,
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
    compare: str | None = None,
    with_source: bool = False,
) -> DatasetReport:
    """End-to-end labeled-dataset eval: load, score, measure, propose bands.

    ``compare`` selects an optional paired A/B: ``"quant"`` (4-bit vs bf16) or
    ``"models"`` (llama vs roberta). In compare mode the headline metrics and
    band proposal read run A, so the report stays comparable to a plain run.
    """
    if compare not in (None, "quant", "models"):
        raise ValueError(f"unknown compare mode {compare!r}")
    samples = load_samples(n, split=split, seed=seed)
    comparison = None
    used_quantized = False
    if compare == "quant":
        comparison = compare_quant(
            samples,
            model=model,
            base=base,
            device=device,
            with_source=with_source,
            reliable_only=reliable_only,
        )
        used_quantized = True
    elif compare == "models":
        comparison = compare_models(
            samples,
            device=device,
            quantize=quantize,
            with_source=with_source,
            reliable_only=reliable_only,
        )
        used_quantized = quantize is not False
    if comparison is not None:
        scored = comparison.scored_a
        used_model, used_device = comparison.model, comparison.device
    else:
        from .engine import EditLens

        engine = EditLens(model=model, base=base, device=device, quantize=quantize)
        used_model, used_device = engine.model_name, engine.device
        used_quantized = engine.quantized
        scored = score_samples(samples, engine=engine, with_source=with_source)

    used = [s for s in scored if s.reliable] if reliable_only else scored
    metrics = compute_dataset_metrics(used, with_source=with_source, seed=seed)
    bands = propose_bands(used, seed=seed)
    return DatasetReport(
        n_total=len(samples),
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
        latency=_latency_summary(used),
        compare=comparison,
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


def bands_artifact(report: DatasetReport) -> dict:
    """Serialize a proposed band table as a reusable calibration artifact.

    The artifact records the cuts *and* their provenance (dataset, split, seed,
    sample size, pinned/interpolated status, CIs) so a consumer can judge
    whether to trust it. Schema consumed by ``band_for(..., bands=...)``.
    """
    return {
        "version": 1,
        "kind": "cx-pangram-bands",
        "model": report.model,
        "cuts": [
            {"lt": cut, "band": name, "pinned": pinned}
            for cut, name, pinned in report.bands.boundaries
        ],
        "top_band": "fully AI",
        "cut_cis": report.bands.cut_cis,
        "degenerate": report.bands.degenerate,
        "provenance": {
            "dataset": DATASET_ID,
            "split": report.split,
            "seed": report.seed,
            "n_used": report.n_used,
            "reliable_only": report.reliable_only,
            "quantized": report.quantized,
        },
    }


def format_markdown_summary(report: DatasetReport) -> str:
    """GFM benchmarks table for the README, with the reproduction command."""
    m = report.metrics
    ci = m.cis

    def cell(v: float, key: str | None = None) -> str:
        s = _fmt(v)
        if key and ci.get(key) and not any(np.isnan(x) for x in ci[key]):
            lo, hi = ci[key]
            s += f" [{_fmt(lo)}, {_fmt(hi)}]"
        return s

    quant = "4-bit" if report.quantized else "unquantized"
    rows = [
        "| model | n | bucket acc | adjacent | spearman | auroc h\\|e | p50 latency |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| {report.model} ({quant}) | {report.n_used} "
        f"| {cell(m.bucket_accuracy, 'bucket_accuracy')} "
        f"| {cell(m.adjacent_accuracy, 'adjacent_accuracy')} "
        f"| {cell(m.spearman_score, 'spearman_score')} "
        f"| {cell(m.auroc_human, 'auroc_human')} "
        f"| {_fmt(report.latency.get('p50_s'))}s |",
    ]
    cmd = (
        f"cx-pangram eval -n {report.n_total} --seed {report.seed} "
        f"--split {report.split} --model {report.model}"
    )
    rows += ["", f"Reproduce: `{cmd}` (bracketed ranges are 95% bootstrap CIs)."]
    return "\n".join(rows)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fmt(x: float | None, places: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{places}f}"


def _describe_line(d: dict) -> str:
    return (
        f"n={d['n']:<5} min={_fmt(d['min'])} q25={_fmt(d['q25'])} "
        f"median={_fmt(d['median'])} mean={_fmt(d['mean'])} "
        f"q75={_fmt(d['q75'])} q90={_fmt(d['q90'])} max={_fmt(d['max'])}"
    )


def _ci_suffix(cis: dict[str, list[float]], key: str) -> str:
    ci = cis.get(key)
    if not ci or any(np.isnan(v) for v in ci):
        return ""
    return f"  [dim][{_fmt(ci[0])}, {_fmt(ci[1])}][/dim]"


def format_dataset_report(report: DatasetReport) -> str:
    """Human-readable dataset eval summary with rich markup.

    Bracketed ranges are seeded 95% bootstrap CIs — at n=200 the headline
    accuracies carry ±several points of sampling noise, so they print alongside
    every number they qualify."""
    m = report.metrics
    cis = m.cis
    lines = [
        f"[bold]dataset eval[/bold]  {report.model} · {report.device} · "
        f"{'4-bit' if report.quantized else 'unquantized'} · "
        f"split={report.split} seed={report.seed}",
        f"scored {report.n_used}/{report.n_total} "
        f"({'reliable only' if report.reliable_only else 'all lengths'})",
        "",
        f"bucket accuracy   {_fmt(m.bucket_accuracy)}{_ci_suffix(cis, 'bucket_accuracy')}",
        f"  argmax variant  {_fmt(m.bucket_accuracy_argmax)}   [dim](mode of weighted softmax; comparable to the paper)[/dim]",
        f"adjacent (±1)     {_fmt(m.adjacent_accuracy)}{_ci_suffix(cis, 'adjacent_accuracy')}",
        f"bucket MAE        {_fmt(m.bucket_mae)}",
        f"spearman score    {_fmt(m.spearman_score)}{_ci_suffix(cis, 'spearman_score')}   [dim](model_score vs cosine_score)[/dim]",
        f"spearman bucket   {_fmt(m.spearman_bucket)}",
        f"auroc human|edited {_fmt(m.auroc_human)}{_ci_suffix(cis, 'auroc_human')}",
        f"auroc edited|full  {_fmt(m.auroc_top)}{_ci_suffix(cis, 'auroc_top')}",
        f"ece (argmax conf) {_fmt(m.ece_argmax)}",
        "",
        "[bold]confusion[/bold] [dim](rows=ground truth, cols=model)[/dim]",
    ]
    for i, row in enumerate(m.confusion):
        lines.append(f"  gt{i}  " + " ".join(f"{c:>5}" for c in row))
    if m.calibration:
        lines.append("")
        lines.append(
            "[bold]calibration[/bold] [dim](score quantile bins: mean score vs mean gt/(n-1))[/dim]"
        )
        for row in m.calibration:
            lines.append(
                f"  n={row['n']:<4} score={_fmt(row['mean_score'])}  gt={_fmt(row['mean_gt'])}"
            )
    lines.append("")
    lines.append("[bold]model_score within each ground-truth bucket[/bold]")
    for b in sorted(m.per_gt_score):
        lines.append(f"  gt{b}  {_describe_line(m.per_gt_score[b])}")
    if m.worst_disagreements:
        lines.append("")
        lines.append("[bold]worst disagreements[/bold]")
        for w in m.worst_disagreements:
            lines.append(
                f"  gt{w['gt_bucket']}→{w['model_bucket']} "
                f"score={_fmt(w['model_score'])}  [dim]{w['preview']}[/dim]"
            )
    if m.source_control:
        lines.append("")
        lines.append(
            f"[bold]source_text control[/bold] (should read human): "
            f"frac in bucket0 = {_fmt(m.source_control['frac_bucket0'])}"
        )
        lines.append("  " + _describe_line(m.source_control["describe"]))
    lines.append("")
    lines.append("[bold]proposed bands[/bold] (model-score scale)")
    key_for = {0: "human", 1: "light", 3: "top"}
    for i, (cut, name, pinned) in enumerate(report.bands.boundaries):
        tag = "pinned" if pinned else "interpolated"
        ci = report.bands.cut_cis.get(key_for.get(i, ""), None)
        ci_txt = (
            f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci and not np.isnan(ci[0]) else ""
        )
        lines.append(f"  < {_fmt(cut)}  {name:<18} [dim]({tag}{ci_txt})[/dim]")
    if report.bands.degenerate:
        lines.append(
            "  [yellow]⚠ monotonicity clamp collapsed a band span; "
            "treat this proposal as unusable[/yellow]"
        )
    if report.latency:
        lines.append("")
        lines.append(
            f"[bold]latency[/bold]  p50={_fmt(report.latency['p50_s'])}s "
            f"p95={_fmt(report.latency['p95_s'])}s · "
            f"{_fmt(report.latency['words_per_s'], 0)} words/s"
        )
    if report.compare:
        qc = report.compare
        lines.append("")
        lines.append(
            f"[bold]{qc.label_a} vs {qc.label_b}[/bold]  "
            f"score MAE={_fmt(qc.score_mae)} "
            f"max Δ={_fmt(qc.max_delta)} bucket-flip={_fmt(qc.bucket_flip_rate)}"
        )
        for label, m, lat in (
            (qc.label_a, qc.metrics_a, qc.latency_a),
            (qc.label_b, qc.metrics_b, qc.latency_b),
        ):
            lines.append(
                f"  {label:<8} acc={_fmt(m.bucket_accuracy)} "
                f"spearman={_fmt(m.spearman_score)} "
                f"p50={_fmt(lat.get('p50_s'))}s · "
                f"{_fmt(lat.get('words_per_s'), 0)} words/s"
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
    for label, score, band in zip(
        report.labels, report.scores, report.bands, strict=True
    ):
        lines.append(f"  {_fmt(score)}  {band:<18} {label}")
    lines.append("")
    lines.append(
        f"human lowest = {report.human_is_min} · rewrite highest = "
        f"{report.rewrite_is_max} · spearman-with-order = {_fmt(report.spearman_order)}"
    )
    return "\n".join(lines)

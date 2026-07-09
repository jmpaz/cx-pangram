"""Shared ref-aware EditLens core for the standalone CLI and the contextualize plugin.

This is the ONE place that resolves refs through contextualize and scores their
authored prose with a single shared EditLens. Both :mod:`cx_pangram.cli` (the
``cx-pangram`` script) and :mod:`cx_pangram.cx_plugin` (the ``contextualize lens``
command) are thin wrappers over :func:`score_refs` and the formatters below.

Detection text per doc follows the prose contract:
- ``prose == ""``      -> skip; the unit declares no authored prose
- ``prose is not None`` -> score that prose; multi-author prose needs ``split``
- ``prose is None``    -> generic-strip fallback over ``content`` so un-migrated
  refs still score (frontmatter, fenced code, heading lines, and ``<image .../>``
  tags removed).

Imports of contextualize and the torch-backed engine are lazy so that ``--help``,
plugin discovery, and raw-text scoring paths stay free of heavy dependencies.
"""

from __future__ import annotations

import json
import re

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCED_CODE_RE = re.compile(
    r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE
)
_HEADING_RE = re.compile(r"^[ \t]*#{1,2}[ \t].*$", re.MULTILINE)
_IMAGE_TAG_RE = re.compile(r"<image\b[^>]*/?>", re.IGNORECASE)


def quiet() -> None:
    """Silence transformers/hf load chatter and progress bars for clean output."""
    import warnings

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    warnings.filterwarnings("ignore")


def _color(score: float) -> str:
    from .bands import band_for

    return band_for(score).color


_UNCOLORED_BANDS = frozenset({"human", "unreliable"})


def _generic_strip(content: str) -> str:
    """Best-effort prose extraction for refs that predate the prose contract."""
    text = _FRONTMATTER_RE.sub("", content or "")
    text = _FENCED_CODE_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _IMAGE_TAG_RE.sub("", text)
    return text.strip()


def _detection_text(doc, split: bool) -> tuple[str | None, str | None]:
    """Resolve a doc to (text, skip_reason); exactly one is non-None."""
    metadata = getattr(doc, "metadata", None) or {}
    if metadata.get("transcript_error"):
        return None, "transcription failed"
    prose = getattr(doc, "prose", None)
    if prose == "":
        return None, "no prose"
    if prose is not None:
        authors = getattr(doc, "prose_authors", None) or []
        if len(authors) > 1 and not split:
            return None, f"{len(authors)} authors (pass --split)"
        return prose, None
    return _generic_strip(getattr(doc, "content", "") or ""), None


def _ref_label(doc) -> str:
    label = getattr(doc, "label", None)
    source = getattr(doc, "source", None)
    return label or source or "<ref>"


def _preview(text: str, width: int = 60) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    return flat[:width]


_EMPTY_ENTRY = {
    "score": None,
    "band": None,
    "bucket": None,
    "n_words": None,
    "n_chunks": None,
    "reliable": None,
    "confidence": None,
    "truncated": None,
    "model": None,
    "calibrated": None,
    "most_ai_chunk": None,
    "chunks": None,
}


def _det_wire(det) -> dict:
    """Detection → result-entry fields; the one mapping between the engine's
    dataclass and the wire dicts every formatter consumes.

    ``chunks`` carry cleaned-word coordinates (``word_start``/``word_end``),
    not offsets into the original string — ``clean_text`` is lossy by design,
    so previews plus cleaned-word spans are what the model actually saw.
    """
    return {
        "score": det.score,
        "band": det.band,
        "bucket": det.bucket,
        "n_words": det.n_words,
        "n_chunks": det.n_chunks,
        "reliable": det.reliable,
        "confidence": det.confidence,
        "truncated": det.truncated,
        "model": det.model,
        "calibrated": det.calibrated,
        "most_ai_chunk": det.most_ai_chunk,
        "chunks": [
            {
                "index": c.index,
                "score": c.score,
                "band": c.band,
                "n_words": c.n_words,
                "word_start": c.word_start,
                "word_end": c.word_end,
                "confidence": c.confidence,
                "truncated": c.truncated,
                "preview": c.preview[:120],
            }
            for c in det.chunks
        ],
    }


def score_refs(
    targets,
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    split: bool = False,
    quantize: bool | None = None,
) -> list[dict]:
    """Resolve refs through contextualize and score their authored prose.

    Returns one result dict per resolved doc with keys ref/source/label plus the
    :func:`_det_wire` fields. Skipped entries carry a reason and null scores;
    scored entries carry a preview of the detection text. All pending docs score
    through one shared engine in a single cross-document batched pass.
    """
    from contextualize import resolve_refs

    from .engine import get_engine

    docs = list(resolve_refs(list(targets), describe_media=False))

    pending: list[tuple[dict, str]] = []
    results: list[dict] = []
    for doc in docs:
        text, skip_reason = _detection_text(doc, split)
        entry = {
            "ref": getattr(doc, "source", None),
            "source": getattr(doc, "source", None),
            "label": _ref_label(doc),
            **_EMPTY_ENTRY,
            "skipped": skip_reason is not None,
            "reason": skip_reason,
            "preview": None,
        }
        results.append(entry)
        if skip_reason is None:
            pending.append((entry, text or ""))

    if pending:
        engine = get_engine(model=model, device=device, base=base, quantize=quantize)
        detections = engine.detect_batch([text for _, text in pending])
        for (entry, text), det in zip(pending, detections, strict=True):
            entry.update(_det_wire(det))
            entry["preview"] = _preview(text)

    return results


def score_text(
    content: str,
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    quantize: bool | None = None,
    label: str = "(text)",
) -> list[dict]:
    """Score a raw string with no ref resolution. Returns a single result dict in
    score_refs's wire shape so the same formatters apply to raw and resolved input."""
    from .engine import get_engine

    engine = get_engine(model=model, device=device, base=base, quantize=quantize)
    det = engine.detect(content)
    return [
        {
            "ref": None,
            "source": None,
            "label": label,
            **_det_wire(det),
            "skipped": False,
            "reason": None,
            "preview": _preview(content),
        }
    ]


def format_jsonl(results: list[dict]) -> str:
    """One JSON object per line; preview is dropped to match the prior wire shape."""
    return "\n".join(json.dumps(_wire(entry)) for entry in results)


def format_json(results: list[dict]) -> str:
    """One JSON array of result objects; preview is dropped to match the prior shape."""
    return json.dumps([_wire(entry) for entry in results])


def _wire(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "preview"}


def _bar(score: float, color: str, width: int = 16) -> str:
    filled = round(score * width)
    return f"[{color}]{'•' * filled}[/{color}][dim]{'·' * (width - filled)}[/dim]"


def format_single(entry: dict) -> str:
    """Detail view for one result."""
    if entry["skipped"]:
        return f"[dim]— skip[/dim]  {entry['label']}  [dim]({entry['reason']})[/dim]"
    score = entry["score"]
    color = _color(score)
    band = entry["band"]
    verdict = band if band in _UNCOLORED_BANDS else f"[{color}]{band}[/{color}]"
    sep_indent = " " * (len(entry["model"]) + 1)
    lines = [
        f"[bold]{score * 100:.0f}% AI[/bold] →  {verdict}",
        _bar(score, color),
        f"[dim]{sep_indent}∵[/dim]",
        f"[dim]{entry['model']} · {entry['n_words']} words, {entry['n_chunks']} chunk(s)[/dim]",
    ]
    if not entry["reliable"]:
        from .engine import MIN_WORDS

        lines.append(
            f"[yellow]⚠ {entry['n_words']} words < {MIN_WORDS}; short-text scores are unreliable[/yellow]"
        )
    if entry.get("calibrated") is False:
        lines.append(
            "[dim]bands tuned for llama; roberta reads high, so treat as a rough gate[/dim]"
        )
    return "\n".join(lines)


def format_human(results: list[dict]) -> str:
    """Band-led one-line-per-ref summary for scanning many refs; rich markup."""
    bandw = max(
        (len("skip" if e["skipped"] else (e["band"] or "")) for e in results),
        default=0,
    )
    uncalibrated = any(
        not e["skipped"] and e.get("calibrated") is False for e in results
    )
    lines: list[str] = []
    for entry in results:
        label = entry["label"]
        mark = (
            "~" if (not entry["skipped"] and entry.get("calibrated") is False) else " "
        )
        if entry["skipped"]:
            lines.append(
                f"{mark} [dim]{'skip'.ljust(bandw)}[/dim]   {label}  [dim]({entry['reason']})[/dim]"
            )
            continue
        score = entry["score"]
        color = _color(score)
        band = (entry["band"] or "").ljust(bandw)
        warn = "" if entry["reliable"] else " [yellow]⚠[/yellow]"
        lines.append(
            f"{mark} [{color}]{band}[/{color}]  [bold]{score * 100:.0f}%[/bold]{warn}   {label}"
        )
    if uncalibrated:
        lines.append(
            "[dim]~ bands tuned for llama; roberta reads high, so treat as a rough gate[/dim]"
        )
    return "\n".join(lines)

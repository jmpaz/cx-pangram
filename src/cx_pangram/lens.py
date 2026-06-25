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
    if score < 0.35:
        return "green"
    if score < 0.55:
        return "yellow"
    if score < 0.75:
        return "dark_orange"
    return "red"


def _generic_strip(content: str) -> str:
    """Best-effort prose extraction for refs that predate the prose contract."""
    text = _FRONTMATTER_RE.sub("", content or "")
    text = _FENCED_CODE_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _IMAGE_TAG_RE.sub("", text)
    return text.strip()


def _detection_text(doc, split: bool) -> tuple[str | None, str | None]:
    """Resolve a doc to (text, skip_reason); exactly one is non-None."""
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


def score_refs(
    targets,
    *,
    model: str = "llama",
    base: str | None = None,
    device: str | None = None,
    split: bool = False,
) -> list[dict]:
    """Resolve refs through contextualize and score their authored prose.

    Returns one result dict per resolved doc with keys ref/source/label/score/
    band/bucket/n_words/reliable/skipped/reason/preview. Skipped entries carry a
    reason and null scores; scored entries carry a preview of the detection text.
    """
    from contextualize import resolve_refs

    from .engine import EditLens

    docs = list(resolve_refs(list(targets)))

    pending: list[tuple[dict, str]] = []
    results: list[dict] = []
    for doc in docs:
        text, skip_reason = _detection_text(doc, split)
        entry = {
            "ref": getattr(doc, "source", None),
            "source": getattr(doc, "source", None),
            "label": _ref_label(doc),
            "score": None,
            "band": None,
            "bucket": None,
            "n_words": None,
            "n_chunks": None,
            "reliable": None,
            "model": None,
            "skipped": skip_reason is not None,
            "reason": skip_reason,
            "preview": None,
        }
        results.append(entry)
        if skip_reason is None:
            pending.append((entry, text or ""))

    if pending:
        engine = EditLens(model=model, device=device, base=base)
        for entry, text in pending:
            det = engine.detect(text)
            entry["score"] = det.score
            entry["band"] = det.band
            entry["bucket"] = det.bucket
            entry["n_words"] = det.n_words
            entry["n_chunks"] = det.n_chunks
            entry["reliable"] = det.reliable
            entry["model"] = det.model
            entry["preview"] = _preview(text)

    return results


def score_text(
    content: str,
    *,
    model: str = "llama",
    base: str | None = None,
    device: str | None = None,
    label: str = "(text)",
) -> list[dict]:
    """Score a raw string with no ref resolution. Returns a single result dict in
    score_refs's wire shape so the same formatters apply to raw and resolved input."""
    from .engine import EditLens

    engine = EditLens(model=model, device=device, base=base)
    det = engine.detect(content)
    return [
        {
            "ref": None,
            "source": None,
            "label": label,
            "score": det.score,
            "band": det.band,
            "bucket": det.bucket,
            "n_words": det.n_words,
            "n_chunks": det.n_chunks,
            "reliable": det.reliable,
            "model": det.model,
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


def _bar(score: float, width: int = 16) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def format_single(entry: dict) -> str:
    """Detail view for one result: verdict line, short bar, blank, then metadata.

    The score reads ``N% AI -> <band>`` so its direction is explicit: 0% is fully
    human, 100% fully AI; the band names where that score sits.
    """
    if entry["skipped"]:
        return f"[dim]— skip[/dim]  {entry['label']}  [dim]({entry['reason']})[/dim]"
    score = entry["score"]
    color = _color(score)
    lines = [
        f"[bold]{score * 100:.0f}%[/bold] AI → [{color}]{entry['band']}[/{color}]",
        f"[{color}]{_bar(score)}[/{color}]",
        "",
        f"[dim]{entry['model']} · {entry['n_words']} words, {entry['n_chunks']} chunk(s)[/dim]",
    ]
    if not entry["reliable"]:
        from .engine import MIN_WORDS

        lines.append(
            f"[yellow]⚠ {entry['n_words']} words < {MIN_WORDS}; short-text scores are unreliable[/yellow]"
        )
    return "\n".join(lines)


def format_human(results: list[dict]) -> str:
    """Band-led one-line-per-ref summary for scanning many refs; rich markup."""
    bandw = max(
        (len("skip" if e["skipped"] else (e["band"] or "")) for e in results),
        default=0,
    )
    lines: list[str] = []
    for entry in results:
        label = entry["label"]
        if entry["skipped"]:
            lines.append(
                f"[dim]{'skip'.ljust(bandw)}[/dim]   {label}  [dim]({entry['reason']})[/dim]"
            )
            continue
        score = entry["score"]
        color = _color(score)
        band = (entry["band"] or "").ljust(bandw)
        warn = "" if entry["reliable"] else " [yellow]⚠[/yellow]"
        lines.append(
            f"[{color}]{band}[/{color}]  [bold]{score * 100:.0f}%[/bold]{warn}   {label}"
        )
    return "\n".join(lines)

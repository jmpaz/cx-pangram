"""Rendering for lens result entries: rich-markup human views and JSON wire forms.

Split from :mod:`cx_pangram.lens` so the scoring core stays presentation-free.
Everything here consumes the wire dicts produced by ``lens.score_refs`` /
``lens.score_text``; colors come from the shared band table in
:mod:`cx_pangram.bands`.
"""

from __future__ import annotations

import json

from .bands import band_for

_UNCOLORED_BANDS = frozenset({"human", "unreliable"})


def _color(score: float) -> str:
    return band_for(score).color


def _wire(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "preview"}


def format_jsonl(results: list[dict]) -> str:
    """One JSON object per line; the top-level preview is dropped, chunk previews
    stay (they are the only text a JSON consumer gets per chunk)."""
    return "\n".join(json.dumps(_wire(entry)) for entry in results)


def format_json(results: list[dict]) -> str:
    """One JSON array of result objects; same field policy as format_jsonl."""
    return json.dumps([_wire(entry) for entry in results])


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
    detail = (
        f"{entry['model']} · {entry['n_words']} words, {entry['n_chunks']} chunk(s)"
    )
    if entry.get("confidence") is not None:
        detail += f" · {entry['confidence'] * 100:.0f}% peaked"
    lines = [
        f"[bold]{score * 100:.0f}% AI[/bold] →  {verdict}",
        _bar(score, color),
        f"[dim]{sep_indent}∵[/dim]",
        f"[dim]{detail}[/dim]",
    ]
    if entry.get("truncated"):
        lines.append(
            "[yellow]⚠ input exceeded the model context in at least one chunk; "
            "overflow was not scored[/yellow]"
        )
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

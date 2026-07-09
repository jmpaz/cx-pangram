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
    chunks = entry.get("chunks") or []
    if len(chunks) > 1:
        lines.append(_heatmap_strip(chunks))
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


_STRIP_GLYPHS = "▁▂▃▄▅▆▇█"


def _heatmap_strip(chunks: list[dict]) -> str:
    """One glyph per chunk — height and color by score — plus a dim legend.

    The strip is the always-on trace of where the AI-edit signal concentrates;
    the full per-chunk table stays behind ``--chunks``.
    """
    glyphs = []
    for c in chunks:
        idx = min(len(_STRIP_GLYPHS) - 1, int(c["score"] * len(_STRIP_GLYPHS)))
        glyphs.append(
            f"[{_color(c['score'])}]{_STRIP_GLYPHS[idx]}[/{_color(c['score'])}]"
        )
    peak = max(chunks, key=lambda c: c["score"])
    return (
        "".join(glyphs)
        + f"  [dim]{len(chunks)} chunks · peak #{peak['index']} "
        + f"{peak['score'] * 100:.0f}% (--chunks for detail)[/dim]"
    )


def format_chunks(entry: dict) -> str:
    """Per-chunk attribution block: bar, score, band, confidence, words, preview."""
    if entry.get("skipped") or not entry.get("chunks"):
        return ""
    lines = []
    for c in entry["chunks"]:
        color = _color(c["score"])
        band = c["band"].ljust(17)
        trunc = " [yellow]⚠trunc[/yellow]" if c.get("truncated") else ""
        lines.append(
            f"  #{c['index']:<3}{_bar(c['score'], color, width=10)} "
            f"[bold]{c['score'] * 100:>3.0f}%[/bold]  [{color}]{band}[/{color}] "
            f"{c['confidence'] * 100:>3.0f}% conf  {c['n_words']:>4}w{trunc}  "
            f"[dim]{c['preview'][:60]}[/dim]"
        )
    return "\n".join(lines)


def format_diff(results: list[dict]) -> str:
    """Side-by-side comparison: one row per target, Δ vs the first scored one."""
    scored = [e for e in results if not e["skipped"] and e["score"] is not None]
    base = scored[0]["score"] if scored else None
    labelw = max((len(e["label"]) for e in results), default=0)
    lines = []
    for entry in results:
        label = entry["label"].ljust(labelw)
        if entry["skipped"]:
            lines.append(f"  [dim]{label}  skip ({entry['reason']})[/dim]")
            continue
        score = entry["score"]
        color = _color(score)
        delta = ""
        if base is not None and entry is not scored[0]:
            d = (score - base) * 100
            sign = "+" if d >= 0 else ""
            delta = f"  [dim]Δ{sign}{d:.0f}[/dim]"
        warn = "" if entry["reliable"] else " [yellow]⚠[/yellow]"
        lines.append(
            f"  {label}  {_bar(score, color, width=12)} "
            f"[bold]{score * 100:>3.0f}%[/bold]  [{color}]{entry['band']}[/{color}]"
            f"{delta}{warn}"
        )
    return "\n".join(lines)


def format_markdown(results: list[dict], *, deltas: bool = False) -> str:
    """GFM table — plain text by construction, safe for machine-adjacent use."""
    scored = [e for e in results if not e["skipped"] and e["score"] is not None]
    base = scored[0]["score"] if scored and deltas else None
    header = "| target | score | band | words |"
    rule = "| --- | --- | --- | --- |"
    if deltas:
        header += " Δ |"
        rule += " --- |"
    header += " note |"
    rule += " --- |"
    rows = [header, rule]
    for entry in results:
        if entry["skipped"]:
            cells = [entry["label"], "—", "—", "—"]
            if deltas:
                cells.append("—")
            cells.append(f"skipped: {entry['reason']}")
        else:
            notes = []
            if not entry["reliable"]:
                notes.append("unreliable")
            if entry.get("truncated"):
                notes.append("truncated")
            cells = [
                entry["label"],
                f"{entry['score'] * 100:.0f}%",
                entry["band"],
                str(entry["n_words"]),
            ]
            if deltas:
                if base is not None and entry is not scored[0]:
                    d = (entry["score"] - base) * 100
                    cells.append(f"{'+' if d >= 0 else ''}{d:.0f}")
                else:
                    cells.append("")
            cells.append(", ".join(notes))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)

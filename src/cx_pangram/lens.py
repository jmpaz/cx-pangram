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

import re
import sys
from contextlib import contextmanager
from pathlib import Path

from .formatters import (
    format_human,
    format_json,
    format_jsonl,
    format_single,
)

__all__ = [
    "score_refs",
    "score_targets",
    "score_text",
    "quiet",
    "output_config",
    "format_human",
    "format_json",
    "format_jsonl",
    "format_single",
]

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


@contextmanager
def output_config(*, verbose: bool = False, quiet_mode: bool = False):
    """Scoped logging/progress policy that restores prior state on exit.

    - transformers/hub *log noise* is off unless ``verbose`` — debug logs are a
      separate concern from progress.
    - HF *progress bars* stay on iff stderr is a TTY and not ``quiet_mode``, so
      a first-run multi-GB checkpoint download is visible interactively but
      never pollutes redirected output (bars write to stderr; stdout stays
      clean for ``--json``).
    - Warning suppression is scoped, not process-global, so a long-lived plugin
      host's warning state is untouched outside the command body.
    """
    import warnings

    from transformers.utils import logging as hf_logging

    try:
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:

        def are_progress_bars_disabled() -> bool:
            return False

        def disable_progress_bars() -> None: ...

        def enable_progress_bars() -> None: ...

    show_bars = sys.stderr.isatty() and not quiet_mode
    prev_verbosity = hf_logging.get_verbosity()
    bars_were_disabled = are_progress_bars_disabled()
    if verbose:
        hf_logging.set_verbosity_info()
    else:
        hf_logging.set_verbosity_error()
    if show_bars:
        enable_progress_bars()
        hf_logging.enable_progress_bar()
    else:
        disable_progress_bars()
        hf_logging.disable_progress_bar()
    with warnings.catch_warnings():
        if not verbose:
            warnings.simplefilter("ignore")
        try:
            yield
        finally:
            hf_logging.set_verbosity(prev_verbosity)
            if bars_were_disabled:
                disable_progress_bars()
            else:
                enable_progress_bars()


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


def score_targets(
    targets,
    *,
    model: str | None = None,
    base: str | None = None,
    device: str | None = None,
    split: bool = False,
    quantize: bool | None = None,
) -> list[dict]:
    """Score a mixed target list in input order: ``-`` reads stdin, an existing
    file path is read directly, anything else resolves as a contextualize ref.

    Local files and stdin need no optional dependency — contextualize is only
    imported when a true ref is present. Everything scores through one shared
    engine in a single cross-document batched pass.
    """
    pending: list[tuple[dict, str]] = []
    results: list[dict] = []

    def add(label: str, source: str | None, text: str | None, reason: str | None):
        entry = {
            "ref": source,
            "source": source,
            "label": label,
            **_EMPTY_ENTRY,
            "skipped": reason is not None,
            "reason": reason,
            "preview": None,
        }
        results.append(entry)
        if reason is None:
            pending.append((entry, text or ""))

    ref_targets = [
        t for t in targets if t != "-" and not (len(t) < 4096 and Path(t).is_file())
    ]
    docs_by_target: dict[str, list] = {}
    if ref_targets:
        from contextualize import resolve_refs

        for t in ref_targets:
            docs_by_target[t] = list(resolve_refs([t], describe_media=False))

    for t in targets:
        if t == "-":
            add("(stdin)", None, sys.stdin.read(), None)
        elif t not in docs_by_target:
            add(t, t, Path(t).read_text(errors="replace"), None)
        else:
            for doc in docs_by_target[t]:
                text, skip_reason = _detection_text(doc, split)
                add(_ref_label(doc), getattr(doc, "source", None), text, skip_reason)

    if pending:
        from .engine import get_engine

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

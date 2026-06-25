"""cx-pangram CLI: ref-aware AI-edit scoring backed by a local EditLens model.

Positional ``TARGET``s are resolved through contextualize and scored as refs.
Raw text is scored directly by the engine, so ``cx-pangram --text ...`` works
without contextualize installed.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import typer
from rich.console import Console

app = typer.Typer(
    add_completion=False,
    help="Local EditLens AI-edit detection (open-pangram). "
    "Pass refs as TARGETS, or raw text via --text / stdin.",
)
console = Console()


def _bar(score: float, width: int = 24) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _render_raw(
    content: str, *, model: str, base: Optional[str], device: Optional[str]
) -> None:
    from . import lens as lens_core
    from .engine import MIN_WORDS, EditLens

    engine = EditLens(model=model, device=device, base=base)
    det = engine.detect(content)
    color = lens_core._color(det.score)
    console.print(
        f"[bold]{det.score * 100:.0f}%[/bold] [{color}]{det.band}[/{color}]  "
        f"[dim]· {det.model} · {det.n_words} words · {det.n_chunks} chunk(s)[/dim]"
    )
    console.print(f"[{color}]{_bar(det.score)}[/{color}] [dim]{det.score:.3f}[/dim]")
    if not det.reliable:
        console.print(
            f"[yellow]⚠ {det.n_words} words < {MIN_WORDS}; short-text scores are unreliable[/yellow]"
        )


@app.command()
def main(
    ctx: typer.Context,
    targets: Optional[List[str]] = typer.Argument(
        None, help="Refs to resolve and score (or use --text / stdin)"
    ),
    text: Optional[str] = typer.Option(
        None, "--text", help="Score this raw text directly (skips ref resolution)"
    ),
    model: str = typer.Option("llama", "--model", "-m", help="llama | roberta"),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        help="Override base model repo (e.g. an ungated Llama-3.2-3B mirror)",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="cuda | cpu | cuda:N"),
    split: bool = typer.Option(
        False, "--split", help="Score multi-author prose instead of skipping it"
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON array of results"
    ),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit one JSON object per line"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show model load logs"),
):
    from . import lens as lens_core

    if not verbose:
        lens_core.quiet()

    raw: Optional[str] = None
    if text is not None:
        raw = text
    elif not targets and not sys.stdin.isatty():
        raw = sys.stdin.read()

    if raw is not None:
        if jsonl:
            typer.echo(
                lens_core.format_jsonl(
                    lens_core.score_text(raw, model=model, base=base, device=device)
                )
            )
            raise typer.Exit()
        if json_out:
            typer.echo(
                lens_core.format_json(
                    lens_core.score_text(raw, model=model, base=base, device=device)
                )
            )
            raise typer.Exit()
        _render_raw(raw, model=model, base=base, device=device)
        raise typer.Exit()

    if not targets:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    try:
        results = lens_core.score_refs(
            targets, model=model, base=base, device=device, split=split
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "contextualize":
            raise typer.BadParameter(
                "scoring refs needs contextualize; install with "
                "`uv tool install cx-pangram[contextualize]` or pass --text for raw input"
            ) from exc
        raise

    if jsonl:
        typer.echo(lens_core.format_jsonl(results))
        raise typer.Exit()
    if json_out:
        typer.echo(lens_core.format_json(results))
        raise typer.Exit()
    console.print(lens_core.format_human(results))

"""cx-pangram CLI: score text for AI-edit extent with a local EditLens model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .engine import MIN_WORDS, EditLens

app = typer.Typer(add_completion=False, help="Local EditLens AI-edit detection (open-pangram).")
console = Console()


def _bar(score: float, width: int = 24) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _color(score: float) -> str:
    if score < 0.10:
        return "green"
    if score < 0.40:
        return "yellow"
    if score < 0.70:
        return "dark_orange"
    return "red"


def _read_clipboard() -> str:
    for cmd in (
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["pbpaste"],
    ):
        if shutil.which(cmd[0]):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            except subprocess.CalledProcessError:
                continue
    raise typer.BadParameter("no clipboard tool found (wl-clipboard, xclip, or pbpaste)")


@app.command()
def main(
    ctx: typer.Context,
    text: Optional[str] = typer.Argument(None, help="Text to score (or use --file / --paste / stdin)"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read text from a file"),
    model: str = typer.Option("roberta", "--model", "-m", help="roberta | llama"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a rendered view"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda | cpu | cuda:N"),
    show_chunks: bool = typer.Option(False, "--chunks", help="Show per-segment scores"),
    paste: bool = typer.Option(False, "--paste", "-p", help="Read text from the clipboard"),
):
    if file is not None:
        content = file.read_text()
    elif text is not None:
        content = text
    elif paste:
        content = _read_clipboard()
    elif not sys.stdin.isatty():
        content = sys.stdin.read()
    else:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    engine = EditLens(model=model, device=device)
    det = engine.detect(content)

    if json_out:
        console.print_json(json.dumps(det.to_dict()))
        raise typer.Exit()

    color = _color(det.score)
    console.print(
        f"[bold]{det.score * 100:.0f}%[/bold] [{color}]{det.band}[/{color}]  "
        f"[dim]· {det.model} · {det.n_words} words · {det.n_chunks} chunk(s)[/dim]"
    )
    console.print(f"[{color}]{_bar(det.score)}[/{color}] [dim]{det.score:.3f}[/dim]")
    if not det.reliable:
        console.print(
            f"[yellow]⚠ {det.n_words} words < {MIN_WORDS}; short-text scores are unreliable[/yellow]"
        )

    if show_chunks and det.n_chunks > 1:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("score", justify="right")
        table.add_column("band")
        table.add_column("words", justify="right")
        table.add_column("preview", overflow="ellipsis", max_width=58)
        for c in det.chunks:
            mark = "  ← most AI" if c.index == det.most_ai_chunk else ""
            table.add_row(str(c.index), f"{c.score:.3f}", c.band + mark, str(c.n_words), c.preview)
        console.print(table)

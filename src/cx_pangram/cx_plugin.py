"""contextualize plugin adapter"""

from __future__ import annotations

import click
from rich.console import Console

PLUGIN_API_VERSION = "1"
PLUGIN_NAME = "pangram"
PLUGIN_PRIORITY = 50

console = Console(highlight=False)


def register_command(root: click.Group) -> None:
    """Mount the ``lens`` command onto the contextualize CLI root group."""

    @root.command(name="lens")
    @click.argument("targets", nargs=-1, required=True)
    @click.option("--model", "-m", default=None, help="auto | llama | roberta")
    @click.option(
        "--base", default=None, help="Override base model repo (e.g. an ungated mirror)"
    )
    @click.option(
        "--json", "json_out", is_flag=True, help="Emit one JSON array of results"
    )
    @click.option("--jsonl", is_flag=True, help="Emit one JSON object per line")
    @click.option(
        "--split",
        is_flag=True,
        help="Score multi-author prose instead of skipping it",
    )
    @click.option("--device", default=None, help="cuda | mps | cpu | cuda:N")
    def lens(
        targets: tuple[str, ...],
        model: str | None,
        base: str | None,
        json_out: bool,
        jsonl: bool,
        split: bool,
        device: str | None,
    ) -> None:
        """Score refs' authored prose for AI-edit extent with a local EditLens."""
        from . import ModelAccessError
        from . import lens as lens_core

        lens_core.quiet()
        try:
            results = lens_core.score_refs(
                targets, model=model, base=base, device=device, split=split
            )
        except ModelAccessError as exc:
            raise click.ClickException(str(exc)) from exc

        if jsonl:
            click.echo(lens_core.format_jsonl(results))
            return
        if json_out:
            click.echo(lens_core.format_json(results))
            return

        console.print(lens_core.format_human(results))

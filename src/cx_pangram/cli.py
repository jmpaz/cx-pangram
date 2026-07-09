"""cx-pangram CLI: ref-aware AI-edit scoring backed by a local EditLens model.

Positional ``TARGET``s are resolved through contextualize and scored as refs.
Raw text is scored directly by the engine, so ``cx-pangram --text ...`` works
without contextualize installed.
"""

from __future__ import annotations

import sys

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

console = Console(highlight=False)


class DefaultCommandGroup(TyperGroup):
    """Route unknown leading tokens to the default ``score`` command.

    A token already naming a subcommand or a group help flag is left untouched; an
    empty invocation, a leading option (``--text``), or a positional ref is dispatched
    to ``score`` so the original single-command surface survives the group restructure.
    """

    default_command = "score"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args:
            args = [self.default_command]
        elif args[0] in self.commands or args[0] in ("--help", "-h"):
            pass
        else:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    add_completion=False,
    cls=DefaultCommandGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Local EditLens AI-edit detection (open-pangram). "
    "Pass refs as TARGETS, or raw text via --text / stdin; `eval` runs the eval harness.",
)


@app.command()
def score(
    ctx: typer.Context,
    targets: list[str] | None = typer.Argument(
        None, help="Refs to resolve and score (or use --text / stdin)"
    ),
    text: str | None = typer.Option(
        None, "--text", help="Score this raw text directly (skips ref resolution)"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="auto | llama | roberta"
    ),
    base: str | None = typer.Option(
        None,
        "--base",
        help="Override base model repo (e.g. an ungated Llama-3.2-3B mirror)",
    ),
    device: str | None = typer.Option(
        None, "--device", help="cuda | mps | cpu | cuda:N"
    ),
    no_quantize: bool = typer.Option(
        False, "--no-quantize", help="Force the unquantized backbone (skip 4-bit)"
    ),
    split: bool = typer.Option(
        False, "--split", help="Score multi-author prose instead of skipping it"
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON array of results"
    ),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit one JSON object per line"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show model load logs"),
):
    """Score refs or raw text for AI-edit extent (default command)."""
    from . import lens as lens_core

    if not verbose:
        lens_core.quiet()

    quantize = False if no_quantize else None

    raw: str | None = None
    if text is not None:
        raw = text
    elif not targets and not sys.stdin.isatty():
        raw = sys.stdin.read()

    if raw is not None:
        results = lens_core.score_text(
            raw, model=model, base=base, device=device, quantize=quantize
        )
        if jsonl:
            typer.echo(lens_core.format_jsonl(results))
            raise typer.Exit()
        if json_out:
            typer.echo(lens_core.format_json(results))
            raise typer.Exit()
        console.print(lens_core.format_single(results[0]))
        raise typer.Exit()

    if not targets:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    try:
        results = lens_core.score_refs(
            targets,
            model=model,
            base=base,
            device=device,
            split=split,
            quantize=quantize,
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
    if len(results) == 1:
        console.print(lens_core.format_single(results[0]))
    else:
        console.print(lens_core.format_human(results))


@app.command(name="eval")
def eval_cmd(
    targets: list[str] | None = typer.Argument(
        None,
        help="Refs for the unlabeled distribution read; omit for the labeled dataset",
    ),
    samples: int = typer.Option(
        200, "-n", "--samples", help="Dataset rows to stream and score"
    ),
    seed: int = typer.Option(0, "--seed", help="Shuffle seed (determinism)"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    model: str | None = typer.Option(
        None, "--model", "-m", help="auto | llama | roberta"
    ),
    base: str | None = typer.Option(None, "--base", help="Override base model repo"),
    device: str | None = typer.Option(
        None, "--device", help="cuda | mps | cpu | cuda:N"
    ),
    no_quantize: bool = typer.Option(
        False, "--no-quantize", help="Force the unquantized backbone (skip 4-bit)"
    ),
    compare_quant: bool = typer.Option(
        False, "--compare-quant", help="Diff 4-bit vs bf16 on the same samples (CUDA)"
    ),
    reliable_only: bool = typer.Option(
        True, "--reliable-only/--all", help="Restrict to >=50-word reliable texts"
    ),
    with_source: bool = typer.Option(
        False,
        "--with-source",
        help="Also score each row's human source_text as a should-read-human control",
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="Run the Ishiguro monotonicity probe instead"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show model load logs"),
):
    """Faithfulness eval: labeled dataset, unlabeled refs, or the smoke gradient."""
    from . import eval as ev
    from . import lens as lens_core

    if not verbose:
        lens_core.quiet()

    quantize = False if no_quantize else None

    def _emit(report, formatter) -> None:
        if json_out:
            import json

            typer.echo(json.dumps(report.to_dict()))
        else:
            console.print(formatter(report))
        raise typer.Exit()

    if smoke:
        _emit(
            ev.smoke_gradient(model=model, base=base, device=device, quantize=quantize),
            ev.format_smoke_report,
        )

    if targets:
        _emit(
            ev.eval_refs(
                targets,
                model=model,
                base=base,
                device=device,
                quantize=quantize,
                reliable_only=reliable_only,
            ),
            ev.format_refs_report,
        )

    _emit(
        ev.eval_dataset(
            n=samples,
            split=split,
            seed=seed,
            model=model,
            base=base,
            device=device,
            quantize=quantize,
            reliable_only=reliable_only,
            compare=compare_quant,
            with_source=with_source,
        ),
        ev.format_dataset_report,
    )

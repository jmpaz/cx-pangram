"""cx-pangram CLI: ref-aware AI-edit scoring backed by a local EditLens model.

Positional ``TARGET``s score in input order: ``-`` reads stdin, an existing file
path is read directly (no optional dependency), anything else resolves through
contextualize. Raw text scores via ``--text`` or piped stdin.

Exit codes: 0 success (gate passed when ``--fail-over`` is set) · 1 runtime
error · 2 usage error · 3 gate tripped · 4 gate indeterminate (nothing scored
reliably).

Heavy imports (torch, transformers) stay inside command bodies so ``--help``,
``--version``, and completions never pay them.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext

import click
import typer
from rich.console import Console
from typer.core import TyperGroup

from . import ModelAccessError

EXIT_GATE_TRIPPED = 3
EXIT_GATE_INDETERMINATE = 4

console = Console(highlight=False)
err_console = Console(highlight=False, stderr=True)

_APP_EPILOG = """Examples:

  cx-pangram essay.md                          score a file

  echo "some text" | cx-pangram --json         raw text, machine output

  cx-pangram diff samples/*.txt                compare an editing gradient

  cx-pangram score -q --fail-over 0.5 *.md     CI gate (exit 3 when over)

Docs: https://github.com/jmpaz/cx-pangram
"""


@contextmanager
def _model_access_guard():
    """Render gated/missing-repo failures as guidance instead of a traceback."""
    try:
        yield
    except ModelAccessError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


def _status(message: str, quiet: bool):
    if sys.stderr.isatty() and not quiet:
        return err_console.status(message)
    return nullcontext()


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


class DefaultCommandGroup(TyperGroup):
    """Route unknown leading tokens to the default ``score`` command.

    A token naming a subcommand or a group-level option (``--help``,
    ``--version``, completion flags) is left untouched; an empty invocation, a
    leading option (``--text``), or a positional ref is dispatched to ``score``
    so the original single-command surface survives the group restructure.
    """

    default_command = "score"

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        root_opts = {
            opt for param in self.params for opt in (*param.opts, *param.secondary_opts)
        }
        root_opts |= {"--help", "-h"}
        if not args:
            args = [self.default_command]
        elif args[0] in self.commands or args[0] in root_opts:
            pass
        else:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultCommandGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Local EditLens AI-edit detection (open-pangram). "
    "Pass files/refs as TARGETS, or raw text via --text / stdin.",
    epilog=_APP_EPILOG,
)


def _version_callback(value: bool):
    if value:
        import importlib.metadata

        typer.echo(f"cx-pangram {importlib.metadata.version('cx-pangram')}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit",
    ),
):
    """Local EditLens AI-edit detection."""


def _resolve_format(fmt: str | None, json_out: bool, jsonl: bool) -> str:
    chosen = [name for flag, name in ((json_out, "json"), (jsonl, "jsonl")) if flag]
    if fmt is not None:
        chosen.append(fmt)
    if len(set(chosen)) > 1:
        raise typer.BadParameter("pick one of --format / --json / --jsonl")
    if chosen and chosen[0] not in ("json", "jsonl", "md"):
        raise typer.BadParameter("--format must be json, jsonl, or md")
    return chosen[0] if chosen else "human"


def _emit_results(
    results: list[dict], fmt: str, *, chunks: bool = False, deltas: bool = False
) -> None:
    from . import formatters

    if fmt == "jsonl":
        typer.echo(formatters.format_jsonl(results))
    elif fmt == "json":
        typer.echo(formatters.format_json(results))
    elif fmt == "md":
        typer.echo(formatters.format_markdown(results, deltas=deltas))
    elif deltas:
        console.print(formatters.format_diff(results))
    elif len(results) == 1:
        console.print(formatters.format_single(results[0]))
        if chunks:
            detail = formatters.format_chunks(results[0])
            if detail:
                console.print(detail)
    else:
        console.print(formatters.format_human(results))
        if chunks:
            for entry in results:
                detail = formatters.format_chunks(entry)
                if detail:
                    console.print(f"[bold]{entry['label']}[/bold]\n{detail}")


def _apply_gate(results: list[dict], threshold: float, quiet: bool) -> int:
    """Exit-code contract for --fail-over: 0 under, 3 over, 4 indeterminate.

    Skipped and unreliable entries never trip the gate — a gate that fails on
    texts the model itself flags as unreliable would page people on noise —
    but a run where *nothing* scored reliably is indeterminate, not a pass.
    """
    scored = [r for r in results if not r["skipped"] and r["reliable"]]
    if not scored:
        if not quiet:
            err_console.print(
                f"[yellow]gate: indeterminate — nothing scored reliably "
                f"(threshold {threshold})[/yellow]"
            )
        return EXIT_GATE_INDETERMINATE
    over = [r for r in scored if r["score"] > threshold]
    if not quiet:
        err_console.print(
            f"[{'red' if over else 'green'}]gate: {len(over)}/{len(scored)} "
            f"over {threshold}[/{'red' if over else 'green'}]"
        )
    return EXIT_GATE_TRIPPED if over else 0


_SCORE_EPILOG = """Examples:

  cx-pangram score essay.md --chunks           per-chunk attribution

  cx-pangram score --fail-over 0.5 posts/*.md  exit 3 if any post reads > 50% AI

  cat draft.txt | cx-pangram score -f md       markdown report from stdin
"""


@app.command(epilog=_SCORE_EPILOG)
def score(
    targets: list[str] | None = typer.Argument(
        None, help="Files, refs, or - for stdin (or use --text / piped stdin)"
    ),
    text: str | None = typer.Option(
        None, "--text", help="Score this raw text directly"
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
    chunks: bool = typer.Option(
        False, "--chunks", help="Show the per-chunk attribution table"
    ),
    fail_over: float | None = typer.Option(
        None,
        "--fail-over",
        min=0.0,
        max=1.0,
        help="Gate: exit 3 if any reliable score exceeds this threshold",
    ),
    fmt: str | None = typer.Option(None, "--format", "-f", help="json | jsonl | md"),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON array of results"
    ),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit one JSON object per line"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="No spinner, progress, or gate summary"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show model load / debug logs"
    ),
):
    """Score files, refs, or raw text for AI-edit extent (default command)."""
    from . import lens as lens_core

    out_fmt = _resolve_format(fmt, json_out, jsonl)
    quantize = False if no_quantize else None

    raw: str | None = None
    if text is not None:
        raw = text
    elif not targets and not _stdin_is_tty():
        raw = sys.stdin.read()

    if raw is None and not targets:
        err_console.print(
            "usage: cx-pangram [TARGETS]... — pass files, refs, --text, or pipe "
            "stdin (try 'cx-pangram --help')"
        )
        raise typer.Exit(2)

    with (
        lens_core.output_config(verbose=verbose, quiet_mode=quiet),
        _model_access_guard(),
        _status("scoring…", quiet),
    ):
        if raw is not None:
            results = lens_core.score_text(
                raw, model=model, base=base, device=device, quantize=quantize
            )
        else:
            try:
                results = lens_core.score_targets(
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
                        "non-file refs need contextualize; install "
                        "cx-pangram[contextualize], or pass file paths / --text"
                    ) from exc
                raise

    _emit_results(results, out_fmt, chunks=chunks)
    if fail_over is not None:
        raise typer.Exit(_apply_gate(results, fail_over, quiet))


_DIFF_EPILOG = """Examples:

  cx-pangram diff samples/*.txt                the bundled human→rewrite gradient

  cx-pangram diff draft-v1.md draft-v2.md      what an editing pass did
"""


@app.command(epilog=_DIFF_EPILOG)
def diff(
    targets: list[str] = typer.Argument(
        ..., help="Two or more files/refs to score side by side"
    ),
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
    split: bool = typer.Option(
        False, "--split", help="Score multi-author prose instead of skipping it"
    ),
    fmt: str | None = typer.Option(None, "--format", "-f", help="json | jsonl | md"),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON array of results"
    ),
    jsonl: bool = typer.Option(False, "--jsonl", help="Emit one JSON object per line"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No spinner or progress"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show model load / debug logs"
    ),
):
    """Compare targets' scores side by side, with deltas against the first."""
    from . import lens as lens_core

    out_fmt = _resolve_format(fmt, json_out, jsonl)
    if len(targets) < 2:
        raise typer.BadParameter("diff needs at least two targets")
    quantize = False if no_quantize else None

    with (
        lens_core.output_config(verbose=verbose, quiet_mode=quiet),
        _model_access_guard(),
        _status("scoring…", quiet),
    ):
        try:
            results = lens_core.score_targets(
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
                    "non-file refs need contextualize; install "
                    "cx-pangram[contextualize] or pass file paths"
                ) from exc
            raise

    _emit_results(results, out_fmt, deltas=True)


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
    compare_models: bool = typer.Option(
        False, "--compare-models", help="Diff llama vs roberta on the same samples"
    ),
    reliable_only: bool = typer.Option(
        True, "--reliable-only/--all", help="Restrict to >=50-word reliable texts"
    ),
    with_source: bool = typer.Option(
        False,
        "--with-source",
        help="Also score each row's human source_text as a should-read-human control",
    ),
    write_bands: str | None = typer.Option(
        None,
        "--write-bands",
        help="Write the proposed band cuts as a calibration artifact (JSON path)",
    ),
    markdown: bool = typer.Option(
        False, "--markdown", help="Emit a GFM benchmarks table instead of the report"
    ),
    smoke: bool = typer.Option(
        False, "--smoke", help="Run the Ishiguro monotonicity probe instead"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No spinner or progress"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show model load / debug logs"
    ),
):
    """Faithfulness eval: labeled dataset, unlabeled refs, or the smoke gradient."""
    from . import eval as ev
    from . import lens as lens_core

    if compare_quant and compare_models:
        raise typer.BadParameter("--compare-quant and --compare-models are exclusive")

    quantize = False if no_quantize else None

    def _emit(report, formatter) -> None:
        if json_out:
            import json

            typer.echo(json.dumps(report.to_dict()))
        else:
            console.print(formatter(report))
        raise typer.Exit()

    with lens_core.output_config(verbose=verbose, quiet_mode=quiet):
        if smoke:
            with _model_access_guard(), _status("running smoke gradient…", quiet):
                report = ev.smoke_gradient(
                    model=model, base=base, device=device, quantize=quantize
                )
            _emit(report, ev.format_smoke_report)

        if targets:
            with _model_access_guard(), _status("scoring refs…", quiet):
                report = ev.eval_refs(
                    targets,
                    model=model,
                    base=base,
                    device=device,
                    quantize=quantize,
                    reliable_only=reliable_only,
                )
            _emit(report, ev.format_refs_report)

        compare = "quant" if compare_quant else "models" if compare_models else None
        with _model_access_guard(), _status(f"evaluating {samples} samples…", quiet):
            report = ev.eval_dataset(
                n=samples,
                split=split,
                seed=seed,
                model=model,
                base=base,
                device=device,
                quantize=quantize,
                reliable_only=reliable_only,
                compare=compare,
                with_source=with_source,
            )
    if write_bands:
        import json
        from pathlib import Path

        Path(write_bands).write_text(json.dumps(ev.bands_artifact(report), indent=2))
        err_console.print(f"[dim]wrote band artifact to {write_bands}[/dim]")
    if markdown:
        typer.echo(ev.format_markdown_summary(report))
        raise typer.Exit()
    _emit(report, ev.format_dataset_report)

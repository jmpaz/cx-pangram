"""DefaultCommandGroup keeps the bare scoring form while adding the eval subcommand.

Exercised through Typer's CliRunner with the heavy seams stubbed, so routing is
observed without importing torch or loading a model.
"""

import pytest

import cx_pangram.cli as cli
import cx_pangram.eval as ev
import cx_pangram.lens as lens

typer_testing = pytest.importorskip("typer.testing")
runner = typer_testing.CliRunner()


def _wire(label, content):
    return {
        "ref": None,
        "source": None,
        "label": label,
        "score": 0.0,
        "band": "human",
        "bucket": 0,
        "n_words": 1,
        "n_chunks": 1,
        "reliable": True,
        "model": "stub",
        "calibrated": True,
        "skipped": False,
        "reason": None,
        "preview": content,
    }


class _Report:
    def __init__(self, marker):
        self.marker = marker

    def to_dict(self):
        return {"marker": self.marker}


@pytest.fixture
def calls(monkeypatch):
    seen = {}
    monkeypatch.setattr(lens, "quiet", lambda: None)

    def score_text(content, **kw):
        seen["score_text"] = content
        return [_wire("(text)", content)]

    def score_refs(targets, **kw):
        seen["score_refs"] = list(targets)
        return [_wire(t, t) for t in targets]

    monkeypatch.setattr(lens, "score_text", score_text)
    monkeypatch.setattr(lens, "score_refs", score_refs)

    def eval_dataset(**kw):
        seen["eval_dataset"] = kw
        return _Report("dataset")

    def eval_refs(targets, **kw):
        seen["eval_refs"] = list(targets)
        return _Report("refs")

    def smoke_gradient(**kw):
        seen["smoke"] = kw
        return _Report("smoke")

    monkeypatch.setattr(ev, "eval_dataset", eval_dataset)
    monkeypatch.setattr(ev, "eval_refs", eval_refs)
    monkeypatch.setattr(ev, "smoke_gradient", smoke_gradient)
    return seen


def test_group_help_lists_eval(calls):
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.output


def test_eval_help_shows_eval_flags(calls):
    result = runner.invoke(cli.app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--compare-quant" in result.output
    assert "--smoke" in result.output


def test_text_flag_routes_to_score(calls):
    result = runner.invoke(cli.app, ["--text", "hello"])
    assert result.exit_code == 0
    assert calls.get("score_text") == "hello"
    assert "eval_dataset" not in calls


def test_positional_routes_to_score(calls):
    result = runner.invoke(cli.app, ["some-ref"])
    assert result.exit_code == 0
    assert calls.get("score_refs") == ["some-ref"]


def test_empty_invocation_routes_to_score(calls):
    result = runner.invoke(cli.app, [], input="")
    assert result.exit_code == 0
    assert "score_text" in calls


def test_eval_without_targets_is_dataset_mode(calls):
    result = runner.invoke(cli.app, ["eval", "--json", "-n", "5", "--seed", "3"])
    assert result.exit_code == 0
    assert "eval_dataset" in calls
    assert calls["eval_dataset"]["n"] == 5
    assert calls["eval_dataset"]["seed"] == 3
    assert "score_refs" not in calls


def test_eval_with_targets_is_refs_mode(calls):
    result = runner.invoke(cli.app, ["eval", "https://are.na/x", "--json"])
    assert result.exit_code == 0
    assert calls.get("eval_refs") == ["https://are.na/x"]


def test_eval_smoke_runs_gradient(calls):
    result = runner.invoke(cli.app, ["eval", "--smoke", "--json"])
    assert result.exit_code == 0
    assert "smoke" in calls


def test_no_quantize_threads_false(calls):
    runner.invoke(cli.app, ["eval", "--no-quantize", "--json"])
    assert calls["eval_dataset"]["quantize"] is False
    runner.invoke(cli.app, ["eval", "--json"])
    assert calls["eval_dataset"]["quantize"] is None


def test_compare_flags_thread_mode_and_are_exclusive(calls):
    runner.invoke(cli.app, ["eval", "--compare-quant", "--json"])
    assert calls["eval_dataset"]["compare"] == "quant"
    runner.invoke(cli.app, ["eval", "--compare-models", "--json"])
    assert calls["eval_dataset"]["compare"] == "models"
    result = runner.invoke(
        cli.app, ["eval", "--compare-quant", "--compare-models", "--json"]
    )
    assert result.exit_code != 0


def test_with_source_threads_through(calls):
    runner.invoke(cli.app, ["eval", "--with-source", "--json"])
    assert calls["eval_dataset"]["with_source"] is True


def test_write_bands_writes_artifact(calls, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(ev, "bands_artifact", lambda report: {"kind": "stub"})
    path = tmp_path / "bands.json"
    result = runner.invoke(cli.app, ["eval", "--json", "--write-bands", str(path)])
    assert result.exit_code == 0
    assert json.loads(path.read_text()) == {"kind": "stub"}


def test_markdown_routes_to_summary(calls, monkeypatch):
    monkeypatch.setattr(ev, "format_markdown_summary", lambda report: "| md table |")
    result = runner.invoke(cli.app, ["eval", "--markdown"])
    assert result.exit_code == 0
    assert "| md table |" in result.output

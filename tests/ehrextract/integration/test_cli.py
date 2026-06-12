"""Tests for the ehrextract CLI: subprocess surface + in-process flag mapping."""

import inspect
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import ehrextract.pipeline as pipeline_mod
from ehrextract import __version__
from ehrextract.providers import AnthropicProvider, HuggingFaceProvider, OpenAIProvider

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "ehrextract", *args],
        env=env,
        capture_output=True,
        text=True,
    )


def test_help_lists_top_level_flags():
    r = _run(["--help"])
    assert r.returncode == 0
    for flag in ("--task", "--provider", "--prompt", "--adapter", "--input",
                 "--output", "--ack-egress", "--max-concurrency", "--api-key-env",
                 "--repetition-penalty", "--trust-remote-code", "--dtype"):
        assert flag in r.stdout
    for dtype_choice in ("bfloat16", "float16", "float32"):
        assert dtype_choice in r.stdout
    # Removed v0.1.1 surface must not reappear.
    for gone in ("--schema-py", "--no-ack-cache", "--no-progress"):
        assert gone not in r.stdout


def test_help_lists_all_three_provider_choices():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "openai" in r.stdout
    assert "anthropic" in r.stdout
    assert "huggingface" in r.stdout


def test_version_prints_0_2_0():
    r = _run(["--version"])
    assert r.returncode == 0
    assert r.stdout.strip() == __version__ == "0.2.0"


def test_missing_required_flags_errors():
    r = _run([])
    assert r.returncode != 0
    assert "required" in r.stderr


# --- in-process flag mapping (CLI -> extract kwargs) ---


def _call_main(monkeypatch, argv: list[str]) -> dict:
    captured: dict = {}

    def fake_extract(notes, task, **kwargs):
        captured["notes"] = notes
        captured["task"] = task
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr("ehrextract.cli.extract", fake_extract)
    from ehrextract.cli import main

    assert main(argv) == 0
    return captured


def test_cli_maps_flags_to_extract_kwargs(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_KEY_ENV", "sk-from-env")
    captured = _call_main(monkeypatch, [
        "--task", "clinical_vars",
        "--provider", "openai",
        "--model", "gpt-4o-mini",
        "--api-key-env", "MY_KEY_ENV",
        "--input", str(tmp_path / "in.jsonl"),
        "--output", str(tmp_path / "out.csv"),
        "--max-new-tokens", "64",
        "--repetition-penalty", "1.15",
        "--ack-egress",
    ])
    assert captured["task"] == "clinical_vars"
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "sk-from-env"
    assert captured["notes"] == tmp_path / "in.jsonl"
    assert captured["output"] == str(tmp_path / "out.csv")
    assert captured["generation"] == {"max_new_tokens": 64, "repetition_penalty": 1.15}
    assert captured["on_egress"] == "silent"
    assert captured["trust_remote_code"] is False  # store_true default
    assert captured["dtype"] == "bfloat16"  # flag default preserves the HF default


def test_cli_generation_flags_default_to_none(monkeypatch, tmp_path):
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["generation"] is None
    assert captured["on_egress"] == "warn"
    assert captured["provider"] == "huggingface"


def test_cli_stdin_mode_passes_inline_text(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin", io.StringIO("inline note text"))
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--input", "-",
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["notes"] == "inline note text"


def test_cli_prompt_flag_reads_file(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("FILE PROMPT", encoding="utf-8")
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--prompt", str(prompt_file),
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["prompt"] == "FILE PROMPT"


# --- A14: new flags, fail-fast exits, summary log, idempotent logging ---


def test_cli_hf_loader_flags(monkeypatch, tmp_path):
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--trust-remote-code",
        "--dtype", "float16",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["trust_remote_code"] is True
    assert captured["dtype"] == "float16"


def test_cli_temperature_zero_is_kept(monkeypatch, tmp_path):
    """Presence checks use `is not None`: an explicit 0 must not be dropped."""
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--temperature", "0",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["generation"] == {"temperature": 0.0}


def test_cli_inline_prompt_passes_through(monkeypatch, tmp_path):
    captured = _call_main(monkeypatch, [
        "--task", "full",
        "--model", "m",
        "--prompt", "Extract everything carefully",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert captured["prompt"] == "Extract everything carefully"


def _main_with_mocked_extract(monkeypatch):
    monkeypatch.setattr("ehrextract.cli.extract", lambda *a, **k: pd.DataFrame())
    from ehrextract.cli import main
    return main


def test_cli_api_key_env_unset_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_VAR", raising=False)
    main = _main_with_mocked_extract(monkeypatch)
    rc = main([
        "--task", "full", "--model", "m",
        "--api-key-env", "DEFINITELY_UNSET_KEY_VAR",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ehrextract: error:" in err
    assert "DEFINITELY_UNSET_KEY_VAR" in err


def test_cli_missing_prompt_path_exits_2(monkeypatch, tmp_path, capsys):
    main = _main_with_mocked_extract(monkeypatch)
    rc = main([
        "--task", "full", "--model", "m",
        "--prompt", str(tmp_path / "nope" / "prompt.txt"),
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ehrextract: error:" in err
    assert "prompt file not found" in err


def test_cli_suffix_only_missing_prompt_exits_2(monkeypatch, tmp_path, capsys):
    """A bare name ending in .md/.txt is path-like even without a separator."""
    main = _main_with_mocked_extract(monkeypatch)
    monkeypatch.chdir(tmp_path)  # ensure the relative name does not exist
    rc = main([
        "--task", "full", "--model", "m",
        "--prompt", "missing_prompt.md",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ])
    assert rc == 2
    assert "prompt file not found" in capsys.readouterr().err


def test_cli_exit_1_when_every_row_is_provider_error(monkeypatch, tmp_path,
                                                     mock_provider_cls):
    class FailingProvider(mock_provider_cls):
        def generate(self, messages, config, json_schema=None):
            err = RuntimeError("bad credentials")
            err.status_code = 401  # non-retryable keeps the test fast (A9)
            raise err

    monkeypatch.setattr(pipeline_mod, "load_provider",
                        lambda name, **kw: FailingProvider())
    inp = tmp_path / "notes.jsonl"
    inp.write_text('{"note_id": "a", "note_text": "first"}\n'
                   '{"note_id": "b", "note_text": "second"}\n')

    from ehrextract.cli import main
    rc = main([
        "--task", "clinical_vars", "--provider", "openai", "--model", "m",
        "--input", str(inp), "--output", str(tmp_path / "r.csv"),
        "--ack-egress",
    ])
    assert rc == 1


def test_cli_logs_end_of_run_summary(monkeypatch, tmp_path, mock_provider_cls, caplog):
    response = json.dumps({
        "tube_feeding": "Y", "oral_feeding": "N",
        "aspiration_risk": "Y", "ni_progressive_or_static": "Static",
    })
    monkeypatch.setattr(pipeline_mod, "load_provider",
                        lambda name, **kw: mock_provider_cls([response, response]))
    inp = tmp_path / "notes.jsonl"
    inp.write_text('{"note_id": "a", "note_text": "first"}\n'
                   '{"note_id": "b", "note_text": "second"}\n')

    from ehrextract.cli import main
    with caplog.at_level(logging.INFO, logger="ehrextract.cli"):
        rc = main([
            "--task", "clinical_vars", "--provider", "openai", "--model", "m",
            "--input", str(inp), "--output", str(tmp_path / "r.csv"),
            "--ack-egress",
        ])
    assert rc == 0
    assert any(r.getMessage() == "done: 2 rows, 2 parsed, 0 provider errors"
               for r in caplog.records)


def test_setup_logging_idempotent_across_main_calls(monkeypatch, tmp_path):
    main = _main_with_mocked_extract(monkeypatch)
    argv = [
        "--task", "full", "--model", "m",
        "--input", str(tmp_path / "in.csv"),
        "--output", str(tmp_path / "out.csv"),
    ]
    assert main(argv) == 0
    assert main(argv) == 0
    stream_handlers = [h for h in logging.getLogger("ehrextract").handlers
                       if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) == 1


# --- provider kwarg signature drift (extract() builds constructor kwargs) ---

_PROVIDER_CASES = {
    "openai": (OpenAIProvider, dict(model="gpt-4o-mini",
                                    base_url="https://api.openai.com/v1",
                                    api_key="sk-test")),
    "anthropic": (AnthropicProvider, dict(model="claude-haiku-4-5", api_key="sk-test")),
    "huggingface": (HuggingFaceProvider, dict(model="m", adapter="/tmp/adapter")),
}


@pytest.mark.parametrize("provider_name", sorted(_PROVIDER_CASES))
def test_provider_kwargs_match_constructor_signature(provider_name, monkeypatch,
                                                     mock_provider_cls):
    """Regression for constructor kwarg drift (e.g. `model` renamed).

    extract() builds a kwargs dict per provider and forwards it to
    load_provider, which passes it to the real constructor. If a provider
    renames a parameter, every CLI invocation silently breaks. We capture
    the kwargs extract() builds and assert each appears in the real
    __init__ signature (no provider is actually constructed).
    """
    real_cls, case = _PROVIDER_CASES[provider_name]
    init_sig = inspect.signature(real_cls.__init__)

    captured: dict = {}

    def fake_load_provider(name, **kwargs):
        captured["kwargs"] = kwargs
        return mock_provider_cls()

    monkeypatch.setattr(pipeline_mod, "load_provider", fake_load_provider)
    from ehrextract import extract
    from ehrextract.schema import FieldSpec, Schema

    schema = Schema(fields=(FieldSpec(name="x", kind="string"),))
    extract(["note"], schema, provider=provider_name, on_egress="silent", **case)

    assert captured["kwargs"], f"extract() built no kwargs for {provider_name}"
    for kwarg_name in captured["kwargs"]:
        assert kwarg_name in init_sig.parameters, (
            f"{real_cls.__name__}.__init__ does not accept {kwarg_name!r} "
            f"that extract() passes for provider={provider_name}. "
            f"Real signature: {init_sig}"
        )


# --- CLI end-to-end through the real extract() path with a mocked provider ---


def test_cli_end_to_end_writes_csv(monkeypatch, tmp_path, mock_provider_cls):
    """CLI calls the library extract() path; only the provider is mocked."""
    response = json.dumps({
        "tube_feeding": "Y",
        "oral_feeding": "N",
        "aspiration_risk": "Y",
        "ni_progressive_or_static": "Static",
    })

    def fake_load_provider(name, **kwargs):
        return mock_provider_cls([response, response])

    monkeypatch.setattr(pipeline_mod, "load_provider", fake_load_provider)

    inp = tmp_path / "notes.jsonl"
    inp.write_text(
        '{"note_id": "a", "note_text": "first"}\n'
        '{"note_id": "b", "note_text": "second"}\n'
    )
    out = tmp_path / "results.csv"

    from ehrextract.cli import main
    rc = main([
        "--task", "clinical_vars",
        "--provider", "openai",
        "--model", "gpt-4o-mini",
        "--input", str(inp),
        "--output", str(out),
        "--ack-egress",
    ])
    assert rc == 0
    loaded = pd.read_csv(out)
    assert list(loaded["note_id"]) == ["a", "b"]
    assert list(loaded["tube_feeding"]) == ["Y", "Y"]
    assert all(loaded["parse_success"])

"""Tests for the stateless egress notice (no ack file, no interactive gate)."""

import threading

import pytest

import ehrextract.providers as providers
from ehrextract.providers import warn_egress


@pytest.fixture(autouse=True)
def fresh_egress_state(monkeypatch, tmp_path):
    """Isolate the per-process warned-destination set and any config dirs."""
    monkeypatch.setattr(providers, "_WARNED_DESTINATIONS", set())
    monkeypatch.delenv("ACK_EGRESS", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path


def test_warn_mode_writes_notice_once_per_destination(capsys):
    warn_egress("api.example.com", mode="warn")
    err = capsys.readouterr().err
    assert "api.example.com" in err
    assert "PHI" in err

    warn_egress("api.example.com", mode="warn")
    assert capsys.readouterr().err == ""


def test_distinct_destinations_each_warn(capsys):
    warn_egress("first.example.com", mode="warn")
    warn_egress("second.example.com", mode="warn")
    err = capsys.readouterr().err
    assert "first.example.com" in err
    assert "second.example.com" in err


def test_silent_mode_never_warns(capsys):
    warn_egress("api.example.com", mode="silent")
    assert capsys.readouterr().err == ""


def test_ack_egress_env_suppresses(monkeypatch, capsys):
    monkeypatch.setenv("ACK_EGRESS", "1")
    warn_egress("api.example.com", mode="warn")
    assert capsys.readouterr().err == ""


def test_no_ack_file_written(fresh_egress_state, capsys):
    warn_egress("api.example.com", mode="warn")
    capsys.readouterr()
    xdg = fresh_egress_state / "xdg"
    assert not xdg.exists() or list(xdg.rglob("*")) == []


# --- A10: notice wording + thread-safe once-per-process dedup ---


def test_notice_says_once_per_process(capsys):
    warn_egress("wording.example.com", mode="warn")
    err = capsys.readouterr().err
    assert "shown once per process" in err
    assert "once per machine" not in err
    assert "once per run" not in err


def test_concurrent_callers_warn_exactly_once(capsys):
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        warn_egress("threads.example.com", mode="warn")

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    err = capsys.readouterr().err
    assert err.count("threads.example.com") == 1

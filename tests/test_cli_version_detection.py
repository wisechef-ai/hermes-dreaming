"""The dreaming provider must advertise the INSTALLED Claude Code version.

Regression 2026-09-24: `_get_cli_version()` hardcoded "2.1.117". After the
default model moved to claude-opus-5-5 (requires Claude Code >= 2.1.280) every
nightly dream failed with HTTP 400 "Claude Code 2.1.117 does not support this
model". The version must be detected, never a stale literal.
"""
from __future__ import annotations

import importlib

import pytest

mod = importlib.import_module("hermes_dreaming.anthropic_oauth_provider")


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("HERMES_DREAMING_CLI_VERSION", raising=False)
    if hasattr(mod, "_cli_version_cache"):
        monkeypatch.setattr(mod, "_cli_version_cache", None, raising=False)


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("HERMES_DREAMING_CLI_VERSION", "9.9.9")
    assert mod._get_cli_version() == "9.9.9"


def test_detects_installed_version(monkeypatch):
    monkeypatch.setattr(mod, "_detect_installed_cli_version", lambda: "2.1.280")
    assert mod._get_cli_version() == "2.1.280"


def test_never_returns_the_stale_literal_when_detection_works(monkeypatch):
    monkeypatch.setattr(mod, "_detect_installed_cli_version", lambda: "3.0.1")
    assert mod._get_cli_version() != "2.1.117"


def test_fallback_is_not_below_opus_5_5_minimum(monkeypatch):
    """With no claude binary at all, the fallback must still be new enough
    for the default model (2.1.280), not the old 2.1.117."""
    monkeypatch.setattr(mod, "_detect_installed_cli_version", lambda: None)
    v = mod._get_cli_version()
    assert tuple(int(x) for x in v.split(".")[:3]) >= (2, 1, 280)


def test_detection_parses_real_binary_output(monkeypatch):
    class R:
        returncode = 0
        stdout = "2.1.280 (Claude Code)\n"

    monkeypatch.setattr(mod.shutil, "which", lambda n: "/fake/claude" if n == "claude" else None)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R())
    assert mod._detect_installed_cli_version() == "2.1.280"

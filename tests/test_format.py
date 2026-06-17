"""Tests for dreaming_format_proposals.py — inline/full/caption modes.

Run: venv/bin/python3 -m pytest tests/test_format.py -q
(script lives in ~/.hermes/scripts, loaded by path here.)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path.home() / ".hermes" / "scripts" / "dreaming_format_proposals.py"


@pytest.fixture(scope="module")
def fmt():
    spec = importlib.util.spec_from_file_location("dreaming_format_proposals", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_artifact(tmp_path: Path, proposals: list[dict]) -> Path:
    art = tmp_path / "20260528T000000Z-test"
    art.mkdir()
    with open(art / "proposals.jsonl", "w") as f:
        for p in proposals:
            f.write(json.dumps(p) + "\n")
    return art


def _big_proposal(kind="memory", path="MEMORY.md"):
    return {
        "target_kind": kind,
        "target_path": path,
        "mode": "append_text",
        "summary": "a big proposal",
        "provenance": ["src.md:1"],
        "proposed_text": "FACT-START " + ("x " * 1500) + "FACT-END",
    }


def test_full_mode_is_untruncated(fmt, tmp_path):
    art = _make_artifact(tmp_path, [_big_proposal()])
    full = fmt.build_full(art)
    # Both ends of a >3000-char entry must survive — no budget in file mode.
    assert "FACT-START" in full
    assert "FACT-END" in full
    assert "more)" not in full  # no truncation marker


def test_inline_mode_respects_2000_limit(fmt, tmp_path):
    # Five big proposals would blow past 2000 if not budgeted.
    art = _make_artifact(tmp_path, [_big_proposal(path=f"M{i}.md") for i in range(5)])
    msg = fmt.build_inline(
        art, live_root="/L", backup_root="/B", archive_root="/A",
    )
    assert len(msg) <= fmt.DISCORD_LIMIT


def test_caption_is_compact_and_names_attachment(fmt, tmp_path):
    art = _make_artifact(tmp_path, [_big_proposal(), _big_proposal(kind="user", path="USER.md")])
    cap = fmt.build_caption(
        art, live_root="/L", backup_root="/B", archive_root="/A",
        attachment_name="dreaming-x.md",
    )
    assert len(cap) < fmt.DISCORD_LIMIT
    assert "dreaming-x.md" in cap
    assert "hermes dreaming -- apply" in cap
    assert "hermes dreaming -- discard" in cap
    # kind counts surfaced
    assert "1×memory" in cap and "1×user" in cap


def test_caption_does_not_embed_full_text(fmt, tmp_path):
    # The caption must NOT contain the big body — that's what the file is for.
    art = _make_artifact(tmp_path, [_big_proposal()])
    cap = fmt.build_caption(art, live_root="/L", backup_root="/B", archive_root="/A")
    assert "FACT-START" not in cap


def test_all_modes_empty_on_no_proposals(fmt, tmp_path):
    art = _make_artifact(tmp_path, [])
    assert fmt.build_full(art) == ""
    assert fmt.build_caption(art, live_root="/L", backup_root="/B", archive_root="/A") == ""
    assert fmt.build_inline(art, live_root="/L", backup_root="/B", archive_root="/A") == ""


def test_full_mode_renders_multiple_entries(fmt, tmp_path):
    p = {
        "target_kind": "memory", "target_path": "MEMORY.md", "mode": "append_text",
        "summary": "multi", "provenance": [],
        "proposed_text": "first entry here\n§\nsecond entry here\n§\nthird entry here",
    }
    art = _make_artifact(tmp_path, [p])
    full = fmt.build_full(art)
    assert "first entry here" in full
    assert "second entry here" in full
    assert "third entry here" in full
    assert "3 entries" in full

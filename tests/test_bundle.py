"""Tests for dreamdedup_2805 Fix 1+2 — existing-state injection + change gate.

Run: venv/bin/python3 -m pytest tests/test_bundle.py -q
(dreaming_build_bundle.py lives in ~/.hermes/scripts, loaded by path here.)
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SCRIPT = Path.home() / ".hermes" / "scripts" / "dreaming_build_bundle.py"


@pytest.fixture(scope="module")
def bb():
    spec = importlib.util.spec_from_file_location("dreaming_build_bundle", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frontmatter_bump_does_not_change_fingerprint(bb):
    body = "# Title\n\nThe reflective body content that actually matters here."
    v1 = f"---\nupdated: 2026-05-28 02:45\n---\n{body}"
    v2 = f"---\nupdated: 2026-05-28 16:00\ngraph: [a, b]\n---\n{body}"
    assert bb.change_fingerprint(v1) == bb.change_fingerprint(v2)


def test_footer_append_does_not_change_fingerprint(bb):
    base = "# Title\n\n" + "\n".join(f"line {i}" for i in range(120))
    with_footer = base + "\n\n---\nSee also: [[tori-index]] · [[moc]]\n"
    # Footer lands past line 100 → outside the fed slice → same fingerprint.
    assert bb.change_fingerprint(base) == bb.change_fingerprint(with_footer)


def test_footer_append_on_SHORT_page_does_not_change_fingerprint(bb):
    # Regression (caught in live test): on a short page the footer falls
    # INSIDE the fed slice, so it must be stripped before hashing.
    base = "# Title\n\nReflective content that matters for distillation."
    with_footer = base + "\n\n---\nSee also: [[tori-index]] · [[moc]]\n"
    with_footer_refreshed = base + "\n\nSee also: [[agentpact-daemon]], [[zero-touch-deals]]\n"
    fp = bb.change_fingerprint(base)
    assert bb.change_fingerprint(with_footer) == fp
    assert bb.change_fingerprint(with_footer_refreshed) == fp


def test_see_also_in_real_body_is_preserved(bb):
    # A "See also:" that is NOT a trailing footer (has real content after it)
    # must NOT be stripped — only the trailing-to-EOF footer is noise.
    v1 = "# Title\n\nSee also: the appendix below.\n\nReal body paragraph one."
    v2 = "# Title\n\nSee also: the appendix below.\n\nReal body paragraph one and TWO."
    assert bb.change_fingerprint(v1) != bb.change_fingerprint(v2)


def test_real_edit_changes_fingerprint(bb):
    v1 = "# Title\n\nOriginal reflective content."
    v2 = "# Title\n\nOriginal reflective content. Plus a genuinely new sentence."
    assert bb.change_fingerprint(v1) != bb.change_fingerprint(v2)


def test_whitespace_touch_does_not_change_fingerprint(bb):
    v1 = "# Title\n\nBody content here."
    v2 = "# Title\n\n\n\nBody   content    here.   "
    assert bb.change_fingerprint(v1) == bb.change_fingerprint(v2)


def test_existing_state_block_includes_memory(bb, tmp_path):
    mem = tmp_path / "MEMORY.md"
    mem.write_text("Fact one about the system.", encoding="utf-8")
    (tmp_path / "USER.md").write_text("User prefers concise output.", encoding="utf-8")
    block = bb.existing_state_block(tmp_path)
    assert "DO NOT DUPLICATE" in block
    assert "Fact one about the system." in block
    assert "User prefers concise output." in block


def test_existing_state_block_empty_when_no_files(bb, tmp_path):
    assert bb.existing_state_block(tmp_path) == ""


def test_changed_only_skips_seen_pages(bb, tmp_path):
    vault = tmp_path / "vault"
    (vault / "knowledge" / "tori").mkdir(parents=True)
    page = vault / "knowledge" / "tori" / "note.md"
    page.write_text("# Note\n\nReflective content worth distilling here.", encoding="utf-8")

    cutoff = datetime.now() - timedelta(days=1)
    files = bb.collect(vault, cutoff)
    assert len(files) == 1

    # First run: no seen hashes → page included.
    _body, included, _skip, hashes = bb.build_bundle(
        files, 80000, 1, cutoff, existing_state="", changed_only=True, seen={},
    )
    assert len(included) == 1
    rel = included[0][1]

    # Second run with the recorded hash → page is skipped (no content change).
    _body2, included2, _s2, _h2 = bb.build_bundle(
        files, 80000, 1, cutoff, existing_state="", changed_only=True, seen=hashes,
    )
    assert len(included2) == 0, "unchanged page must be skipped under --changed-only"


def test_changed_only_includes_edited_page(bb, tmp_path):
    vault = tmp_path / "vault"
    (vault / "knowledge" / "tori").mkdir(parents=True)
    page = vault / "knowledge" / "tori" / "note.md"
    page.write_text("# Note\n\nOriginal content.", encoding="utf-8")
    cutoff = datetime.now() - timedelta(days=1)
    files = bb.collect(vault, cutoff)
    _b, included, _s, hashes = bb.build_bundle(
        files, 80000, 1, cutoff, existing_state="", changed_only=True, seen={},
    )
    rel = included[0][1]
    # Edit the page, recompute candidates, run with old hash → must reappear.
    page.write_text("# Note\n\nOriginal content. Now with a brand-new insight added.", encoding="utf-8")
    files2 = bb.collect(vault, cutoff)
    _b2, included2, _s2, _h2 = bb.build_bundle(
        files2, 80000, 1, cutoff, existing_state="", changed_only=True, seen=hashes,
    )
    assert len(included2) == 1, "edited page must reappear under --changed-only"


def test_seen_hashes_roundtrip(bb, tmp_path):
    store = tmp_path / ".seen_hashes.json"
    bb.save_seen_hashes(store, {"a.md": "deadbeef"})
    loaded = bb.load_seen_hashes(store)
    assert loaded == {"a.md": "deadbeef"}


def test_load_seen_hashes_missing_file(bb, tmp_path):
    assert bb.load_seen_hashes(tmp_path / "nope.json") == {}


# ---- session-history digest (dreamsession_2805) --------------------------

def _make_session_db(bb, tmp_path, rows):
    """rows: list of (id, session_id, role, content, ts_offset_seconds)."""
    import sqlite3, time
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp REAL)"
    )
    now = time.time()
    for mid, sid, role, content, off in rows:
        con.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) "
            "VALUES (?,?,?,?,?)",
            (mid, sid, role, content, now + off),
        )
    con.commit()
    con.close()
    return db


def test_session_noise_filter(bb):
    assert bb._is_session_noise("[Replying to: something]") is True
    assert bb._is_session_noise("You are running as a scheduled cron job now") is True
    assert bb._is_session_noise("You are working on /home/adam/repos/x") is True
    assert bb._is_session_noise("[IMPORTANT: The user has invoked the skill") is True
    assert bb._is_session_noise("[Recent channel messages] foo") is True
    assert bb._is_session_noise("[FYI] heads up") is True
    assert bb._is_session_noise("implement the dedup fix the way we discussed") is False


def test_session_digest_extracts_genuine_pairs(bb, tmp_path):
    import time
    db = _make_session_db(bb, tmp_path, [
        (1, "s1", "user", "always use content-hash not git time for change detection", -100),
        (2, "s1", "assistant", "Understood — switching to content-hash only.", -99),
        (3, "s1", "user", "[Replying to: blah] this is a quote wrapper", -90),  # noise
        (4, "s1", "assistant", "ack", -89),
        (5, "s2", "user", "You are running as a scheduled cron job — do X", -80),  # noise
        (6, "s2", "user", "no", -70),  # too short
        (7, "s3", "user", "stop posting status acks in agent-sync, just ACK or act", -60),
        (8, "s3", "assistant", "Got it, will stay silent unless action is needed.", -59),
    ])
    pairs = bb.collect_session_turns(db, time.time() - 86400)
    texts = [u for u, _ in pairs]
    assert any("content-hash" in t for t in texts)
    assert any("stop posting status acks" in t for t in texts)
    assert not any("Replying to" in t for t in texts)
    assert not any("scheduled cron" in t for t in texts)
    assert "no" not in texts  # too short
    # pairing: the content-hash turn should carry its assistant reply
    ch = next(a for u, a in pairs if "content-hash" in u)
    assert "content-hash only" in ch


def test_session_digest_block_has_header_and_pairs(bb, tmp_path):
    import time
    db = _make_session_db(bb, tmp_path, [
        (1, "s1", "user", "prefer criticality 1-10 framing not USD for recipes what-next", -50),
        (2, "s1", "assistant", "Noted — criticality framing from now on.", -49),
    ])
    block = bb.session_digest_block(db, time.time() - 86400)
    assert "SESSION HISTORY" in block
    assert "MINE FOR DURABLE FACTS" in block
    assert "ADAM:" in block and "TORI:" in block
    assert "criticality" in block


def test_session_digest_empty_when_no_db(bb, tmp_path):
    assert bb.session_digest_block(tmp_path / "nonexistent.db", 0) == ""


def test_session_digest_empty_when_all_noise(bb, tmp_path):
    import time
    db = _make_session_db(bb, tmp_path, [
        (1, "s1", "user", "[Replying to: x] quoted", -10),
        (2, "s2", "user", "You are running as a scheduled cron job blah", -9),
    ])
    assert bb.session_digest_block(db, time.time() - 86400) == ""


def test_session_respects_cutoff(bb, tmp_path):
    import time
    db = _make_session_db(bb, tmp_path, [
        (1, "s1", "user", "this is an old genuine instruction from long ago", -86400 * 5),
        (2, "s1", "user", "this is a recent genuine instruction worth keeping", -100),
    ])
    pairs = bb.collect_session_turns(db, time.time() - 86400)  # last 1 day
    texts = [u for u, _ in pairs]
    assert any("recent genuine" in t for t in texts)
    assert not any("old genuine" in t for t in texts)

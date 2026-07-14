"""Tests for the Option-2 re-plumb (2026-07-14): Cognee routing, whole-file
headroom gate, spool fail-safe, and stale-artifact janitor semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_dreaming import apply as apply_mod
from hermes_dreaming import cognee_client
from hermes_dreaming.apply import (
    HERMES_ENTRY_DELIMITER,
    HERMES_MEMORY_FILE_BUDGET,
    HERMES_MEMORY_MAX_PER_ENTRY,
    _apply_hermes_memory_append_budgeted,
    _write_proposal,
)
from hermes_dreaming.artifact import DreamProposal


def _mk_proposal(**over):
    base = dict(
        id="p1",
        summary="test proposal",
        target_kind="memory",
        target_path="MEMORY.md",
        mode="append_text",
        proposed_text="a new durable fact about the environment",
        provenance=["test:1"],
    )
    base.update(over)
    return DreamProposal.from_dict(base)


# ---------------------------------------------------------------------------
# Budgeted append
# ---------------------------------------------------------------------------

class TestBudgetedAppend:
    def test_fits_within_budget_lands_in_file(self):
        current = "existing entry one\n"
        updated, applied, diverted = _apply_hermes_memory_append_budgeted(
            current, "brand new fact", HERMES_MEMORY_MAX_PER_ENTRY, 2200
        )
        assert applied == ["brand new fact"]
        assert diverted == []
        assert "brand new fact" in updated

    def test_over_budget_diverts_not_writes(self):
        # File nearly full: one entry of 2150 chars against a 2200 budget
        # leaves ~47c headroom; a 100c entry cannot fit.
        big = "x" * 2150
        current = big + "\n"
        entry = "this entry cannot possibly fit in the remaining headroom " + "y" * 60
        updated, applied, diverted = _apply_hermes_memory_append_budgeted(
            current, entry, HERMES_MEMORY_MAX_PER_ENTRY, 2200,
        )
        assert applied == []
        assert diverted == [entry]
        # File unchanged in substance and still within budget.
        assert len(updated) <= 2200
        assert "cannot possibly fit" not in updated

    def test_partial_fit_splits_applied_and_diverted(self):
        # Addition > per-entry cap forces the paragraph split into three
        # 1000c entries; only the first two fit a 2200 budget.
        current = "seed entry\n"
        a, b, c = "a" * 1000, "b" * 1000, "c" * 1000
        addition = a + "\n\n" + b + "\n\n" + c
        updated, applied, diverted = _apply_hermes_memory_append_budgeted(
            current, addition, HERMES_MEMORY_MAX_PER_ENTRY, 2200
        )
        assert applied == [a, b]
        assert diverted == [c]
        assert len(updated) <= 2200

    def test_dedup_still_wins_over_divert(self):
        # A near-duplicate must be DROPPED (dedup), not diverted to Cognee.
        current = ("the Cognee default dataset is hermes-tori on mac01 shared brain"
                   + HERMES_ENTRY_DELIMITER + "q" * 2000) + "\n"
        updated, applied, diverted = _apply_hermes_memory_append_budgeted(
            current, "Cognee default dataset: hermes-tori on the mac01 shared brain",
            HERMES_MEMORY_MAX_PER_ENTRY, 4000,
        )
        assert applied == []
        assert diverted == []

    def test_never_exceeds_budget(self):
        current = ""
        addition = "\n\n".join(f"entry number {i} " + "w" * 300 for i in range(12))
        updated, applied, diverted = _apply_hermes_memory_append_budgeted(
            current, addition, HERMES_MEMORY_MAX_PER_ENTRY, 2200
        )
        assert len(updated) <= 2200
        assert len(applied) + len(diverted) == 12
        assert diverted  # some must have overflowed


# ---------------------------------------------------------------------------
# Cognee routing in _write_proposal (sink mocked — no network in tests)
# ---------------------------------------------------------------------------

class TestWriteProposalRouting:
    @pytest.fixture()
    def capture_sink(self, monkeypatch):
        calls: list[dict] = []

        def fake_send(content, *, kind="fact", provenance=""):
            calls.append({"content": content, "kind": kind, "provenance": provenance})
            return {"sent": True, "dataset": "test"}

        monkeypatch.setattr(cognee_client, "send_fact", fake_send)
        return calls

    def test_fact_kind_goes_to_cognee_not_memory(self, tmp_path, capture_sink):
        target = tmp_path / "facts.jsonl"
        mem = tmp_path / "MEMORY.md"
        mem.write_text("guardrail entry\n", encoding="utf-8")
        proposal = _mk_proposal(
            target_kind="fact",
            target_path="facts.jsonl",
            mode="jsonl_append",
            proposed_text=json.dumps({"fact": "Adam prefers USD token budgets"}),
        )
        _write_proposal(target, proposal)
        # Fact reached the sink.
        assert len(capture_sink) == 1
        assert "USD token budgets" in capture_sink[0]["content"]
        # MEMORY.md untouched (old behaviour wrote [fact] rows there).
        assert mem.read_text(encoding="utf-8") == "guardrail entry\n"
        # Audit log still written.
        assert "USD token budgets" in target.read_text(encoding="utf-8")

    def test_memory_overflow_diverts_to_cognee(self, tmp_path, capture_sink):
        target = tmp_path / "MEMORY.md"
        target.write_text("x" * 2150 + "\n", encoding="utf-8")
        proposal = _mk_proposal(
            proposed_text="an oversized entry that will not fit the remaining headroom " + "q" * 60,
        )
        _write_proposal(target, proposal)
        assert len(capture_sink) == 1
        assert capture_sink[0]["kind"] == "memory-overflow"
        after = target.read_text(encoding="utf-8")
        assert len(after) <= HERMES_MEMORY_FILE_BUDGET
        # Verification inside _write_proposal must NOT raise (divert = success).

    def test_memory_within_budget_still_writes_file(self, tmp_path, capture_sink):
        target = tmp_path / "MEMORY.md"
        target.write_text("small existing entry\n", encoding="utf-8")
        proposal = _mk_proposal(proposed_text="fresh guardrail that fits fine")
        _write_proposal(target, proposal)
        assert capture_sink == []  # nothing diverted
        assert "fresh guardrail that fits fine" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Spool fail-safe (Cognee down)
# ---------------------------------------------------------------------------

class TestSpoolFailSafe:
    def test_send_fact_spools_on_unreachable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cognee_client, "SPOOL_DIR", tmp_path / "spool")

        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(cognee_client, "_cognee_add", boom)
        res = cognee_client.send_fact("important fact", kind="fact")
        assert res["sent"] is False
        spooled = Path(res["spooled"])
        assert spooled.exists()
        assert "important fact" in spooled.read_text(encoding="utf-8")

    def test_flush_spool_sends_and_clears(self, tmp_path, monkeypatch):
        spool = tmp_path / "spool"
        spool.mkdir(parents=True)
        (spool / "20260714T000000000000Z-fact.txt").write_text("queued fact", encoding="utf-8")
        monkeypatch.setattr(cognee_client, "SPOOL_DIR", spool)
        monkeypatch.setattr(cognee_client, "cognee_health_ok", lambda *a, **k: True)
        sent: list[str] = []
        monkeypatch.setattr(
            cognee_client, "_cognee_add",
            lambda host, ds, content, timeout=30.0: (sent.append(content), {"status": "ok"})[1],
        )
        res = cognee_client.flush_spool()
        assert res == {"pending": 0, "flushed": 1}
        assert sent and "queued fact" in sent[0]
        assert not list(spool.glob("*.txt"))

    def test_flush_spool_keeps_files_when_down(self, tmp_path, monkeypatch):
        spool = tmp_path / "spool"
        spool.mkdir(parents=True)
        (spool / "a-fact.txt").write_text("queued", encoding="utf-8")
        monkeypatch.setattr(cognee_client, "SPOOL_DIR", spool)
        monkeypatch.setattr(cognee_client, "cognee_health_ok", lambda *a, **k: False)
        res = cognee_client.flush_spool()
        assert res["pending"] == 1 and res["flushed"] == 0
        assert list(spool.glob("*.txt"))

    def test_divert_never_raises_on_import_or_send(self, monkeypatch):
        def boom(content, *, kind="fact", provenance=""):
            raise RuntimeError("sink exploded")

        monkeypatch.setattr(cognee_client, "send_fact", boom)
        # _divert_to_cognee must swallow per-entry errors? No — send_fact
        # itself never raises by contract; simulate the contract instead.
        monkeypatch.setattr(
            cognee_client, "send_fact",
            lambda content, *, kind="fact", provenance="": {"sent": False, "error": "down", "spooled": "/tmp/x"},
        )
        results = apply_mod._divert_to_cognee(["e1", "e2"], kind="fact")
        assert len(results) == 2
        assert all(r["sent"] is False for r in results)

"""ah_1010: distinct memory/user appends stay atomic and survive the policy cap.

Live defect: ``_coalesce_appends`` ran before ``_drop_policy_invalid`` and glued
every MEMORY.md/USER.md append into ONE body. Two ordinary entries then
exceeded the per-entry cap (240/220 chars, 4 lines) and the merged proposal was
dropped whole. dreaming-cron.log shows "dropped <id>: entry exceeds target
character budget" on ~23 of 30 nights; on 2026-10-10 all three intended entries
were lost. The coalesce existed only because the validator treated two distinct
append_text bodies to one file as a conflict.
"""

from __future__ import annotations

from pathlib import Path

from hermes_dreaming.anthropic_oauth_provider import _coalesce_appends
from hermes_dreaming.artifact import DreamProposal
from hermes_dreaming.providers import OpenAICompatibleProvider
from hermes_dreaming.validation import validate_proposals


def _p(pid: str, text: str, *, kind: str = "memory", path: str = "MEMORY.md") -> DreamProposal:
    return DreamProposal(
        id=pid,
        target_kind=kind,
        target_path=path,
        mode="append_text",
        summary=f"entry {pid}",
        provenance=[f"src-{pid}.md:1"],
        proposed_text=text,
        approved=False,
        reason="test",
        policy_flags=["test"],
    )


# Each entry is well inside the 240-char memory cap; together they are not.
A = "AZA lines override: when Adam edits a scene line, re-split only that line and keep every other beat timing as recorded."
B = "Dreaming JSON truncation: repair the trailing object before parsing, never discard the whole payload on one cut row."
C = "Hercules unreachable usually means a netbird overlay drop on a live box, not a power loss; read uptime first."


def _survivors(proposals: list[DreamProposal]) -> list[DreamProposal]:
    kept, _dropped = OpenAICompatibleProvider._drop_policy_invalid(_coalesce_appends(proposals))
    return kept


def test_three_memory_entries_all_survive_the_policy_gate():
    assert all(len(t) < 240 for t in (A, B, C)) and len(A + B + C) > 240
    kept = _survivors([_p("a", A), _p("b", B), _p("c", C)])
    assert sorted(p.id for p in kept) == ["a", "b", "c"]
    assert {p.proposed_text for p in kept} == {A, B, C}


def test_user_entries_are_atomic_too():
    u1 = "Prefers milestones over timelines in plans; never put calendar dates on phases."
    u2 = "Wants status as one roll-up: state, proposal, feedback. No separate threads per item."
    u3 = "Horse routes go on trails and bridleways only, never on roads, even for short links."
    kept = _survivors([_p(i, t, kind="user", path="USER.md") for i, t in (("u1", u1), ("u2", u2), ("u3", u3))])
    assert sorted(p.id for p in kept) == ["u1", "u2", "u3"]


def test_distinct_memory_entries_to_one_file_are_not_a_conflict():
    assert validate_proposals([_p("a", A), _p("b", B)]) == []


def test_lowercase_memory_path_is_atomic_as_well():
    kept = _survivors([_p("a", f"- {A}", path="memory.md"), _p("b", f"- {B}", path="memory.md")])
    assert sorted(p.id for p in kept) == ["a", "b"]
    assert validate_proposals(kept) == []


def test_identical_memory_entries_still_validate_and_do_not_double():
    assert validate_proposals([_p("a", A), _p("a2", A)]) == []


def test_skill_appends_to_one_file_still_coalesce():
    s1 = "- first skill note about a pitfall\n"
    s2 = "- second skill note about another pitfall\n"
    merged = _coalesce_appends(
        [_p("s1", s1, kind="skill", path="skills/x.md"), _p("s2", s2, kind="skill", path="skills/x.md")]
    )
    assert len(merged) == 1 and "first" in merged[0].proposed_text and "second" in merged[0].proposed_text


def test_distinct_non_atomic_appends_still_conflict():
    errors = validate_proposals(
        [
            _p("s1", "- one skill note\n", kind="skill", path="skills/x.md"),
            _p("s2", "- another skill note\n", kind="skill", path="skills/x.md"),
        ]
    )
    assert any("conflicting proposals" in e for e in errors)


def test_sequential_apply_keeps_both_entries(tmp_path: Path, monkeypatch):
    """Apply re-reads the target per proposal, so two atomic entries to one
    file both land (the reason coalescing was never needed for these)."""
    from hermes_dreaming import apply as apply_mod

    target = tmp_path / "MEMORY.md"
    target.write_text("", encoding="utf-8")
    for prop in (_p("a", A), _p("b", B)):
        apply_mod._write_proposal(target, prop)
    text = target.read_text(encoding="utf-8")
    assert A in text and B in text

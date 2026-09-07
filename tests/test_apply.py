from __future__ import annotations

from pathlib import Path

import pytest

from hermes_dreaming.artifact import DreamArtifact, DreamProposal, load_artifact, write_artifact
from hermes_dreaming.apply import apply_artifact, discard_artifact, DreamApplyError


def _artifact(tmp_path: Path, proposal: DreamProposal) -> tuple[Path, DreamArtifact]:
    artifact = DreamArtifact(
        artifact_id="artifact-apply",
        created_at="2026-05-25T12:00:00Z",
        provider="offline-marker",
        status="validated",
        workspace_root=str(tmp_path),
        source_roots=[str(tmp_path / "sources")],
        report="# Report",
        sources=[],
        proposals=[proposal],
    )
    artifact_dir = tmp_path / "artifact"
    write_artifact(artifact, artifact_dir)
    return artifact_dir, artifact


def test_apply_appends_memory_and_writes_backup(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    live_root.mkdir()
    memory = live_root / "memory.md"
    memory.write_text("# MEMORY\n\n- Existing note\n", encoding="utf-8")

    proposal = DreamProposal(
        id="proposal-memory",
        target_kind="memory",
        target_path="memory.md",
        mode="append_text",
        summary="append memory note",
        provenance=["sessions/1.md:1"],
        proposed_text="- Keep updates short and concrete.",
        approved=True,
    )
    artifact_dir, artifact = _artifact(tmp_path, proposal)
    backup_root = tmp_path / "backups"

    result = apply_artifact(artifact_dir, live_root=live_root, backup_root=backup_root, approve_all=True)

    assert result.status == "applied"
    assert memory.read_text(encoding="utf-8").strip().endswith("- Keep updates short and concrete.")
    assert (backup_root / "memory.md").exists()
    assert load_artifact(artifact_dir).status == "applied"


def test_apply_requires_approval(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    live_root.mkdir()
    (live_root / "memory.md").write_text("# MEMORY\n", encoding="utf-8")

    proposal = DreamProposal(
        id="proposal-memory",
        target_kind="memory",
        target_path="memory.md",
        mode="append_text",
        summary="append memory note",
        provenance=["sessions/1.md:1"],
        proposed_text="- Keep updates short and concrete.",
        approved=False,
    )
    artifact_dir, _artifact_result = _artifact(tmp_path, proposal)

    with pytest.raises(DreamApplyError):
        apply_artifact(artifact_dir, live_root=live_root, backup_root=tmp_path / "backups", approve_all=False)


def test_discard_moves_artifact_to_archive_without_live_mutation(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    live_root.mkdir()
    memory = live_root / "memory.md"
    memory.write_text("# MEMORY\n", encoding="utf-8")

    proposal = DreamProposal(
        id="proposal-memory",
        target_kind="memory",
        target_path="memory.md",
        mode="append_text",
        summary="append memory note",
        provenance=["sessions/1.md:1"],
        proposed_text="- Keep updates short and concrete.",
        approved=True,
    )
    artifact_dir, artifact = _artifact(tmp_path, proposal)
    archive_root = tmp_path / "archive"

    discarded_path = discard_artifact(artifact_dir, archive_root=archive_root)

    assert discarded_path.exists()
    assert not artifact_dir.exists()
    assert memory.read_text(encoding="utf-8") == "# MEMORY\n"
    assert load_artifact(discarded_path).status == "discarded"


def _two_proposal_artifact(tmp_path: Path) -> Path:
    """An artifact with one MEMORY and one USER proposal, both un-approved."""
    live_root = tmp_path / "live"
    live_root.mkdir()
    (live_root / "MEMORY.md").write_text("seed mem\n", encoding="utf-8")
    (live_root / "USER.md").write_text("seed user\n", encoding="utf-8")

    mem = DreamProposal(
        id="prop-memory",
        target_kind="memory",
        target_path="MEMORY.md",
        mode="append_text",
        summary="memory fact",
        provenance=["sessions/1.md:1"],
        proposed_text="A brand new memory fact about deployment topology.",
        approved=False,
    )
    usr = DreamProposal(
        id="prop-user",
        target_kind="user",
        target_path="USER.md",
        mode="append_text",
        summary="user pref",
        provenance=["sessions/1.md:2"],
        proposed_text="PREF a brand new user preference worth recording here.",
        approved=False,
    )
    artifact = DreamArtifact(
        artifact_id="artifact-partial",
        created_at="2026-06-02T00:00:00Z",
        provider="offline-marker",
        status="validated",
        workspace_root=str(live_root),
        source_roots=[str(tmp_path / "sources")],
        report="# Report",
        sources=[],
        proposals=[mem, usr],
    )
    artifact_dir = tmp_path / "artifact"
    write_artifact(artifact, artifact_dir)
    return artifact_dir


def test_partial_apply_reports_partially_applied_and_persists_flags(tmp_path: Path) -> None:
    """Approving ONLY the USER proposal must NOT stamp the artifact 'applied'.

    Regression for the June-1 silent-partial-apply bug: status was
    unconditionally set to 'applied' and the per-proposal 'applied' flag
    was a runtime-only attribute dropped by asdict(), so the review surface
    could not see that the MEMORY proposal never landed.
    """
    live_root = tmp_path / "live"
    artifact_dir = _two_proposal_artifact(tmp_path)

    result = apply_artifact(
        artifact_dir,
        live_root=live_root,
        backup_root=tmp_path / "backups",
        approve_ids=["prop-user"],
    )

    # Status reflects the gap, not a blanket "applied".
    assert result.status == "partially_applied"

    # The USER proposal landed; the MEMORY proposal did not.
    assert "brand new user preference" in (live_root / "USER.md").read_text(encoding="utf-8")
    assert "deployment topology" not in (live_root / "MEMORY.md").read_text(encoding="utf-8")

    # Per-proposal flags persisted to disk (the asdict() round-trip fix).
    reloaded = load_artifact(artifact_dir)
    by_id = {p.id: p for p in reloaded.proposals}
    assert by_id["prop-user"].applied is True
    assert by_id["prop-memory"].applied is False
    assert reloaded.status == "partially_applied"
    assert reloaded.applied_at is not None


def test_full_apply_reports_applied_and_flags_all(tmp_path: Path) -> None:
    """Approving every proposal stamps 'applied' and flags both proposals."""
    live_root = tmp_path / "live"
    artifact_dir = _two_proposal_artifact(tmp_path)

    result = apply_artifact(
        artifact_dir,
        live_root=live_root,
        backup_root=tmp_path / "backups",
        approve_all=True,
    )

    assert result.status == "applied"
    reloaded = load_artifact(artifact_dir)
    assert all(p.applied for p in reloaded.proposals)
    assert reloaded.status == "applied"


def test_apply_jsonl_append_handles_literal_backslash_n_separated_rows(tmp_path: Path) -> None:
    """dreamescape_0824 regression — apply path, not just validation.

    Same escaping-bug payload as the validation regression test: multiple
    JSON objects joined by the literal two-char text ``\\n`` instead of a
    real newline. Must apply cleanly, land both rows in facts.jsonl, and
    pass the post-write verification loop (which independently re-parses
    proposed_text and used to share the same splitlines() bug).
    """
    live_root = tmp_path / "live"
    live_root.mkdir()

    proposal = DreamProposal(
        id="proposal-fact-backslash-n",
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary="two facts joined by literal backslash-n (escaping bug)",
        provenance=["sessions/1.md:1"],
        proposed_text='{"fact": "AGENTPACT apps/web is Fastify"}\\n{"fact": "LOOPSKILL has no bundle install verb"}',
        approved=True,
    )
    artifact_dir, _artifact_result = _artifact(tmp_path, proposal)

    result = apply_artifact(
        artifact_dir, live_root=live_root, backup_root=tmp_path / "backups", approve_all=True
    )

    assert result.status == "applied"
    facts_text = (live_root / "facts.jsonl").read_text(encoding="utf-8")
    assert "AGENTPACT apps/web is Fastify" in facts_text
    assert "LOOPSKILL has no bundle install verb" in facts_text
    # Two distinct rows landed, not one mangled row.
    assert len([line for line in facts_text.splitlines() if line.strip()]) == 2


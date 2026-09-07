from __future__ import annotations

from pathlib import Path

import pytest

from hermes_dreaming.artifact import DreamArtifact, DreamProposal
from hermes_dreaming.validation import validate_artifact


def _artifact_with(proposal: DreamProposal, tmp_path: Path) -> DreamArtifact:
    return DreamArtifact(
        artifact_id="artifact-001",
        created_at="2026-05-25T12:00:00Z",
        provider="offline-marker",
        status="staged",
        workspace_root=str(tmp_path),
        source_roots=[str(tmp_path)],
        report="# Report",
        sources=[],
        proposals=[proposal],
    )


def test_validate_rejects_path_traversal(tmp_path: Path) -> None:
    proposal = DreamProposal(
        id="proposal-bad-path",
        target_kind="memory",
        target_path="../outside.md",
        mode="append_text",
        summary="bad path",
        provenance=["sessions/1.md:1"],
        proposed_text="- Do not write outside roots.",
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors
    assert any("outside" in error.lower() for error in errors)


def test_validate_rejects_secret_like_content(tmp_path: Path) -> None:
    proposal = DreamProposal(
        id="proposal-secret",
        target_kind="memory",
        target_path="memory.md",
        mode="append_text",
        summary="contains secret-like string",
        provenance=["sessions/1.md:1"],
        proposed_text="- api_key = 'ghp_1234567890abcdef1234567890abcdef1234'",
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors
    assert any("secret" in error.lower() for error in errors)


def test_validate_rejects_missing_provenance(tmp_path: Path) -> None:
    proposal = DreamProposal(
        id="proposal-no-provenance",
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary="missing provenance",
        provenance=[],
        proposed_text='{"type": "preference", "value": "concise"}',
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors
    assert any("provenance" in error.lower() for error in errors)


def _fact(pid: str, payload: str) -> DreamProposal:
    return DreamProposal(
        id=pid,
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary=f"fact {pid}",
        provenance=["sessions/1.md:1"],
        proposed_text=payload,
        approved=True,
    )


def test_validate_allows_distinct_jsonl_rows_to_same_file(tmp_path: Path) -> None:
    """Two different facts to facts.jsonl are additive, not a conflict.

    Regression: a live superqwen run on 2026-09-07 produced 3 distinct facts and
    the whole artifact went `invalid` with "conflicting proposals target the same
    path 'facts.jsonl'" — one row per fact is exactly what jsonl_append is for.
    """
    artifact = _artifact_with(_fact("f1", '{"k": "a"}'), tmp_path)
    artifact.proposals.append(_fact("f2", '{"k": "b"}'))

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert not [e for e in errors if "conflicting proposals" in e], errors


def test_validate_still_rejects_conflicting_append_text_same_path(tmp_path: Path) -> None:
    """The append_text/replace_text conflict rule must survive the jsonl carve-out."""
    def _mem(pid: str, text: str) -> DreamProposal:
        return DreamProposal(
            id=pid, target_kind="memory", target_path="memory.md", mode="append_text",
            summary=pid, provenance=["sessions/1.md:1"], proposed_text=text, approved=True,
        )
    artifact = _artifact_with(_mem("m1", "- one"), tmp_path)
    artifact.proposals.append(_mem("m2", "- two"))

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert any("conflicting proposals" in e for e in errors)


def test_validate_accepts_real_newline_jsonl(tmp_path: Path) -> None:
    """Baseline: multiple JSON objects joined by an actual newline byte."""
    proposal = DreamProposal(
        id="proposal-multirow-newline",
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary="two facts, real newline separated",
        provenance=["sessions/1.md:1"],
        proposed_text='{"fact": "one"}\n{"fact": "two"}',
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors == []


def test_validate_accepts_literal_backslash_n_separated_jsonl(tmp_path: Path) -> None:
    """dreamescape_0824 regression.

    2026-08-24: Opus double-escaped the row separator inside the outer
    envelope JSON, so ``proposed_text`` (after the outer JSON decode) is
    two well-formed JSON objects joined by the literal two-char text
    ``\\n`` — NOT a real newline byte. The old `.splitlines()` walker saw
    one "line" containing both objects and failed with "Extra data",
    marking the whole artifact invalid and silently dropping a day of
    dreaming output. Must validate clean.
    """
    proposal = DreamProposal(
        id="proposal-multirow-literal-backslash-n",
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary="two facts, literal backslash-n separated (escaping bug)",
        provenance=["sessions/1.md:1"],
        proposed_text='{"fact": "one"}\\n{"fact": "two"}',
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors == []


def test_validate_still_rejects_genuinely_malformed_jsonl_row(tmp_path: Path) -> None:
    """The tolerant walker must not swallow real garbage between good rows."""
    proposal = DreamProposal(
        id="proposal-malformed-row",
        target_kind="fact",
        target_path="facts.jsonl",
        mode="jsonl_append",
        summary="one good row, one genuinely broken row",
        provenance=["sessions/1.md:1"],
        proposed_text='{"fact": "one"}\nnot json at all',
        approved=True,
    )
    artifact = _artifact_with(proposal, tmp_path)

    errors = validate_artifact(artifact, live_root=tmp_path)

    assert errors
    assert any("malformed jsonl payload" in error.lower() for error in errors)

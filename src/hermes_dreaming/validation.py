from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Iterable

from .artifact import DreamArtifact, DreamProposal, SourceSnapshot, VALID_MODES, VALID_PRIORITY_LEVELS, VALID_RISK_LEVELS, VALID_TARGET_KINDS
from .policy import _extract_jsonl_rows, evaluate_live_op, evaluate_proposal, target_path_is_allowed

SECRET_PATTERNS = [
    re.compile(r"\b(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{10,})\b"),
    re.compile(r"\b(api[_-]?key|secret|password|token|bearer)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.]{8,}", re.IGNORECASE),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
]


# Target kinds whose append_text proposals are atomic entries (one fact each,
# capped per entry by policy.TARGET_POLICY). Distinct entries never conflict
# and are never merged.
ATOMIC_ENTRY_KINDS = frozenset({"memory", "user"})


def _secret_like(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _safe_relative_path(path_text: str) -> bool:
    path = PurePosixPath(path_text.replace("\\", "/"))
    if path.is_absolute():
        return False
    return all(part not in {"..", ""} for part in path.parts)


def _proposal_errors(proposal: DreamProposal) -> list[str]:
    errors: list[str] = []
    if proposal.target_kind not in VALID_TARGET_KINDS:
        errors.append(f"proposal {proposal.id} has unsupported target kind {proposal.target_kind!r}")
    if proposal.mode not in VALID_MODES:
        errors.append(f"proposal {proposal.id} has unsupported mode {proposal.mode!r}")
    if not proposal.summary.strip():
        errors.append(f"proposal {proposal.id} is missing a summary")
    if not proposal.provenance:
        errors.append(f"proposal {proposal.id} is missing provenance")
    if not proposal.target_path.strip() or not _safe_relative_path(proposal.target_path):
        errors.append(f"proposal {proposal.id} has an unsafe target path {proposal.target_path!r}")
    elif not target_path_is_allowed(proposal.target_kind, proposal.target_path):
        errors.append(
            f"proposal {proposal.id} target path {proposal.target_path!r} is not allowed for {proposal.target_kind!r}"
        )
    if proposal.confidence < 0.0 or proposal.confidence > 1.0:
        errors.append(f"proposal {proposal.id} has invalid confidence {proposal.confidence!r}")
    if proposal.risk not in VALID_RISK_LEVELS:
        errors.append(f"proposal {proposal.id} has invalid risk {proposal.risk!r}")
    if proposal.priority not in VALID_PRIORITY_LEVELS:
        errors.append(f"proposal {proposal.id} has invalid priority {proposal.priority!r}")
    if any(not str(flag).strip() for flag in proposal.policy_flags):
        errors.append(f"proposal {proposal.id} has an empty policy flag")
    if _secret_like(proposal.summary) or _secret_like(proposal.proposed_text) or _secret_like(proposal.snippet) or _secret_like(proposal.reason) or _secret_like(proposal.source_quote):
        errors.append(f"proposal {proposal.id} contains secret-like content")
    if proposal.mode == "jsonl_append":
        # Accept one JSON object, true JSONL (one per line), or objects joined by
        # the literal backslash-n artifact (dreamescape_0824). Strict superset of
        # a single json.loads — genuinely malformed rows still fail.
        if not proposal.proposed_text.strip():
            errors.append(f"proposal {proposal.id} has empty JSONL payload")
        else:
            objects, err = _extract_jsonl_rows(proposal.proposed_text)
            if err:
                errors.append(f"proposal {proposal.id} {err}")
            elif not objects or not all(isinstance(o, dict) for o in objects):
                errors.append(f"proposal {proposal.id} must serialize a JSON object for jsonl_append")
    return errors


def validate_proposals(proposals: Iterable[DreamProposal]) -> list[str]:
    errors: list[str] = []
    seen_targets: dict[str, str] = {}
    seen_keys: dict[str, str] = {}
    for proposal in proposals:
        errors.extend(_proposal_errors(proposal))
        decision = evaluate_proposal(proposal)
        if not decision.ok:
            errors.append(f"proposal {proposal.id}: {decision.error}")
            continue

        # Distinct jsonl_append rows to the same file are additive, not conflicting
        # (each row is its own JSON object). So are distinct memory/user entries
        # (ah_1010): each is one atomic append under a per-entry cap, so merging
        # them is what made them fail the cap. Other append_text bodies collide.
        target_key = (
            f"{proposal.target_path}\x00{decision.normalized_text}"
            if proposal.mode == "jsonl_append"
            or (proposal.mode == "append_text" and proposal.target_kind in ATOMIC_ENTRY_KINDS)
            else proposal.target_path
        )
        existing = seen_targets.get(target_key)
        if existing is not None and existing != decision.normalized_text:
            errors.append(f"conflicting proposals target the same path {proposal.target_path!r}")
        else:
            seen_targets[target_key] = decision.normalized_text

        if proposal.idempotence_key:
            existing_key = seen_keys.get(proposal.idempotence_key)
            if existing_key is not None and existing_key != decision.normalized_text:
                errors.append(f"conflicting payloads share the same idempotence key for {proposal.idempotence_key!r}")
            else:
                seen_keys[proposal.idempotence_key] = decision.normalized_text
    return errors


def validate_source_snapshots(sources: Iterable[SourceSnapshot]) -> list[str]:
    errors: list[str] = []
    for source in sources:
        if _secret_like(source.content):
            errors.append(f"source {source.path} contains secret-like content")
    return errors


def validate_artifact(artifact: DreamArtifact, *, live_root: Path | str) -> list[str]:
    errors: list[str] = []
    live_root = Path(live_root)

    if not artifact.proposals:
        errors.append("artifact contains no proposals")

    errors.extend(validate_source_snapshots(artifact.sources))
    errors.extend(validate_proposals(artifact.proposals))

    if not live_root.exists():
        errors.append(f"live root does not exist: {live_root}")

    return errors


def validate_memory_op(
    *,
    op: str,
    target: str,
    old_text: str | None,
    new_text: str | None,
    reason: str,
    sources: Iterable[str],
    score: float,
    supersession_confidence: float = 0.0,
) -> list[str]:
    """Validate a live memory mutation before score gating and write attempts."""
    errors: list[str] = []

    if target not in VALID_TARGET_KINDS:
        errors.append(f"unsupported target kind {target!r}")
    if op not in {"add", "replace", "remove"}:
        errors.append(f"unsupported operation {op!r}")
    if not reason.strip():
        errors.append("reason is required")
    sources_list = [str(source) for source in sources]
    if not sources_list or any(not source.strip() for source in sources_list):
        errors.append("sources are required")

    if _secret_like(reason):
        errors.append("reason contains secret-like content")
    if old_text and _secret_like(old_text):
        errors.append("old_text contains secret-like content")
    if new_text and _secret_like(new_text):
        errors.append("new_text contains secret-like content")

    if op in {"add", "replace"}:
        if new_text is None or not new_text.strip():
            errors.append(f"{op} requires new_text")
        elif not new_text.strip().startswith("-"):
            errors.append(f"{op} new_text must be a bullet entry")
    if op in {"replace", "remove"} and (old_text is None or not old_text.strip()):
        errors.append(f"{op} requires old_text")

    if op == "replace" and supersession_confidence < 0.75:
        errors.append("replace requires supersession confidence >= 0.75")
    if op == "remove" and supersession_confidence < 0.85:
        errors.append("remove requires supersession confidence >= 0.85")

    if score < 0.0 or score > 1.0:
        errors.append(f"score must be between 0.0 and 1.0, got {score!r}")

    live_decision = evaluate_live_op(
        op=op,
        target=target,
        old_text=old_text,
        new_text=new_text,
        reason=reason,
        sources=sources_list,
    )
    if not live_decision.ok:
        errors.append(live_decision.error)

    return errors

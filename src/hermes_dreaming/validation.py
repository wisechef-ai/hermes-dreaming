from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Iterable

from .artifact import DreamArtifact, DreamProposal, VALID_MODES, VALID_TARGET_KINDS

SECRET_PATTERNS = [
    re.compile(r"\b(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{10,})\b"),
    re.compile(r"\b(api[_-]?key|secret|password|token|bearer)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.]{8,}", re.IGNORECASE),
]

# Generic 32+ hex string check — only fires when NOT obviously a documented
# digest. We strip lines that look like sha256/digest/hash labels or that
# sit inside backtick-fenced code/inline code, then check the residue.
_HEX32_RE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")
_HEX_LABEL_RE = re.compile(
    r"(sha\d*|md5|digest|hash|commit|fingerprint|checksum|sri|content-id|etag|"
    r"job_id|run_id|trace_id|span_id|request_id|message_id|uuid|guid|"
    r"oid|tx|txid|block|nonce|node_id|chunk_id|artifact_id)",
    re.IGNORECASE,
)


def _looks_like_documented_digest(text: str, match_start: int) -> bool:
    """Heuristic: a 32+ hex string is treated as a documented digest (not a
    secret) when EITHER (a) the same line contains a digest-context word like
    sha256/digest/commit/job_id, or (b) the match sits inside backticks on
    the same line (a code span — markdown convention for documented hashes)."""
    # Walk back to start-of-line, forward to end-of-line.
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_start)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    if _HEX_LABEL_RE.search(line):
        return True
    # Backtick check: count backticks before the match on this line. If odd,
    # we're inside an inline code span.
    rel_pos = match_start - line_start
    if line[:rel_pos].count("`") % 2 == 1:
        return True
    return False


def _secret_like(text: str) -> bool:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    for m in _HEX32_RE.finditer(text):
        if not _looks_like_documented_digest(text, m.start()):
            return True
    return False


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
    if _secret_like(proposal.summary) or _secret_like(proposal.proposed_text):
        errors.append(f"proposal {proposal.id} contains secret-like content")
    if proposal.mode == "jsonl_append":
        # Accept either (a) a single JSON object, or (b) multiple JSON
        # objects, one per non-empty line (true JSONL). Opus and other
        # LLMs naturally produce (b) when asked to append several facts.
        rows = [r for r in proposal.proposed_text.splitlines() if r.strip()]
        if not rows:
            errors.append(f"proposal {proposal.id} has empty JSONL payload")
        else:
            for idx, row in enumerate(rows, 1):
                try:
                    parsed = json.loads(row)
                except json.JSONDecodeError:
                    errors.append(
                        f"proposal {proposal.id} row {idx}/{len(rows)} has malformed JSONL payload"
                    )
                    break
                else:
                    if not isinstance(parsed, dict):
                        errors.append(
                            f"proposal {proposal.id} row {idx}/{len(rows)} must serialize a JSON object for jsonl_append"
                        )
                        break
    return errors


def validate_artifact(artifact: DreamArtifact, *, live_root: Path | str) -> list[str]:
    errors: list[str] = []
    live_root = Path(live_root)

    if not artifact.proposals:
        errors.append("artifact contains no proposals")

    seen_targets: dict[str, str] = {}
    for source in artifact.sources:
        if _secret_like(source.content):
            errors.append(f"source {source.path} contains secret-like content")

    for proposal in artifact.proposals:
        errors.extend(_proposal_errors(proposal))
        # jsonl_append targets are intentionally multi-row — multiple
        # proposals appending different rows to the same .jsonl is the
        # design, not a conflict. Only enforce uniqueness for append_text
        # (which writes the same target with a single body).
        if proposal.mode == "jsonl_append":
            continue
        existing = seen_targets.get(proposal.target_path)
        if existing is not None and existing != proposal.proposed_text:
            errors.append(f"conflicting proposals target the same path {proposal.target_path!r}")
        else:
            seen_targets[proposal.target_path] = proposal.proposed_text

    if not live_root.exists():
        errors.append(f"live root does not exist: {live_root}")

    return errors


def validate_proposals(proposals: Iterable[DreamProposal]) -> list[str]:
    errors: list[str] = []
    seen_targets: dict[str, str] = {}
    for proposal in proposals:
        errors.extend(_proposal_errors(proposal))
        existing = seen_targets.get(proposal.target_path)
        if existing is not None and existing != proposal.proposed_text:
            errors.append(f"conflicting proposals target the same path {proposal.target_path!r}")
        else:
            seen_targets[proposal.target_path] = proposal.proposed_text
    return errors

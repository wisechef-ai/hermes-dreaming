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


# dreamescape_0824 (2026-08-24) — Opus over-escaping fallback.
# ---------------------------------------------------------------------------
# Over-fit 3 (see references/upstream-validator-overfits.md) taught the
# validator to walk proposed_text `.splitlines()` and parse each non-empty
# line as one JSON object — correct when the model joins rows with a REAL
# newline byte (0x0A). Observed 2026-08-24: Opus double-escaped the row
# separator inside the outer envelope JSON, emitting the *literal two-char
# text* `\n` (backslash + 'n') between rows instead of an actual newline.
# `.splitlines()` doesn't split on that, so N rows arrive concatenated on
# one "line" and `json.loads` fails with "Extra data" at the second object's
# start — the whole artifact gets marked invalid and a day of dreaming is
# lost, even though every individual JSON object is well-formed.
#
# Fix: use `json.JSONDecoder.raw_decode` to walk proposed_text and pull out
# consecutive JSON objects regardless of what sits between them (real
# newline, whitespace, or the literal backslash-n artifact). This is a
# strict superset of the old `.splitlines()` behaviour — well-formed
# single-newline-separated JSONL still round-trips identically — and it
# tolerates the escaping bug without silently accepting garbage: any byte
# that isn't whitespace or the literal `\n` artifact between two valid JSON
# objects still fails decoding and reports the same malformed-payload error.
_JSONL_SEPARATOR_WS = " \t\r\n"


def _skip_jsonl_separator(text: str, idx: int) -> int:
    while idx < len(text):
        if text[idx] in _JSONL_SEPARATOR_WS:
            idx += 1
        elif text[idx : idx + 2] == "\\n":
            idx += 2
        else:
            break
    return idx


def _extract_jsonl_rows(text: str) -> tuple[list[object], str | None]:
    """Parse ``text`` as zero or more JSON objects.

    Returns ``(objects, None)`` on success or ``(partial_objects, error)``
    where ``error`` is a human-readable message identifying the row index
    (1-based) that failed to parse. Tolerates real newlines, whitespace, and
    the literal backslash-n separator artifact between objects.
    """
    decoder = json.JSONDecoder()
    objects: list[object] = []
    idx = _skip_jsonl_separator(text, 0)
    row_num = 0
    while idx < len(text):
        row_num += 1
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            return objects, f"row {row_num} has malformed JSONL payload"
        objects.append(obj)
        idx = _skip_jsonl_separator(text, end)
    return objects, None


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
        # Accept (a) a single JSON object, (b) multiple JSON objects one per
        # non-empty line (true JSONL), or (c) multiple JSON objects joined by
        # the literal backslash-n artifact (dreamescape_0824) — a strict
        # superset that still catches genuinely malformed payloads.
        if not proposal.proposed_text.strip():
            errors.append(f"proposal {proposal.id} has empty JSONL payload")
        else:
            objects, err = _extract_jsonl_rows(proposal.proposed_text)
            if err:
                errors.append(f"proposal {proposal.id} {err}")
            elif not objects:
                errors.append(f"proposal {proposal.id} has empty JSONL payload")
            else:
                for idx, obj in enumerate(objects, 1):
                    if not isinstance(obj, dict):
                        errors.append(
                            f"proposal {proposal.id} row {idx}/{len(objects)} must serialize a JSON object for jsonl_append"
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

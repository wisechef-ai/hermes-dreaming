from __future__ import annotations

"""Target-aware policy helpers for staged and live Hermes Dreaming writes.

This module keeps the policy concerns separate from I/O:
- normalize proposals before hashing
- compute durable idempotence keys
- enforce per-target capacity and anti-slop limits
- classify stale/superseded fact payloads

The policy surface is intentionally boring. If the text is sloppy, too large,
or pointed at the wrong target, we reject it here before anything touches live
state.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any, Iterable

from .artifact import DreamProposal, VALID_TARGET_KINDS

POLICY_VERSION = "2026-06-02"

TARGET_POLICY: dict[str, dict[str, int]] = {
    "memory": {"max_chars": 240, "max_lines": 4, "total_chars": 4000},
    "user": {"max_chars": 220, "max_lines": 4, "total_chars": 4000},
    "skill": {"max_chars": 900, "max_lines": 24, "total_chars": 12000},
    "fact": {"max_chars": 320, "max_lines": 12, "total_chars": 12000},
}

RUN_POLICY = {
    "max_changes": 3,
    "max_adds": 1,
    "max_new_chars": 250,
    "max_targets": 3,
}

_GENERIC_SLOP_RE = re.compile(r"\b(important|various|misc(?:ellaneous)?|stuff|things|etc\.?|general)\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{10,})\b"
)


@dataclass(slots=True)
class PolicyDecision:
    ok: bool
    target_kind: str
    mode: str
    idempotence_key: str
    policy_version: str
    normalized_text: str
    normalized_old_text: str | None
    lifecycle: str = "active"
    capacity_ok: bool = True
    capacity_reason: str = ""
    error: str = ""
    warnings: list[str] | None = None


@dataclass(slots=True)
class LivePolicyDecision:
    ok: bool
    target: str
    op: str
    idempotence_key: str
    policy_version: str
    normalized_old_text: str | None
    normalized_new_text: str | None
    lifecycle: str = "active"
    capacity_ok: bool = True
    capacity_reason: str = ""
    error: str = ""
    warnings: list[str] | None = None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    collapsed: list[str] = []
    blank_run = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not blank_run:
                collapsed.append("")
            blank_run = True
            continue
        collapsed.append(stripped)
        blank_run = False
    return "\n".join(collapsed)


def normalize_provenance(provenance: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in provenance:
        item = str(entry).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def fingerprint_provenance(provenance: Iterable[str]) -> str:
    cleaned = sorted(normalize_provenance(provenance))
    return "|".join(cleaned) if cleaned else "-"


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


def _canonical_json_text(text: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Canonicalise a jsonl_append payload of 1..N JSON objects.

    Returns the canonical newline-joined rows and the FIRST object (used for
    lifecycle classification). Multi-row payloads are legal JSONL — see
    validation._extract_jsonl_rows for the separator tolerance (dreamescape_0824).
    """
    objects, err = _extract_jsonl_rows(text)
    if err or not objects or not all(isinstance(o, dict) for o in objects):
        return None, None
    canonical = "\n".join(json.dumps(o, sort_keys=True, ensure_ascii=False) for o in objects)
    return canonical, objects[0]


_SKILL_TARGET_RE = re.compile(r"^skills/[a-z0-9][a-z0-9_-]{0,63}\.md$")


def target_path_is_allowed(target_kind: str, target_path: str) -> bool:
    path = PurePosixPath(target_path.replace("\\", "/"))
    path_text = path.as_posix()
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        return False
    # Hermes' live files are UPPERCASE (~/.hermes/memories/MEMORY.md, USER.md);
    # upstream's quickstart uses lowercase. Both are the same target — apply.py
    # already prefers an existing uppercase file (v0.3.0). Compare case-folded.
    if target_kind == "memory":
        return path_text.lower() == "memory.md"
    if target_kind == "user":
        return path_text.lower() == "user.md"
    if target_kind == "fact":
        return path_text == "facts.jsonl"
    if target_kind == "skill":
        return bool(_SKILL_TARGET_RE.fullmatch(path_text))
    return False


def _target_path_is_safe(target_kind: str, target_path: str) -> bool:
    return target_path_is_allowed(target_kind, target_path)


def _entry_too_sloppy(target_kind: str, text: str, target_path: str = "") -> str:
    lines = text.splitlines()
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if target_kind in {"memory", "user"}:
        # Hermes' MEMORY.md / USER.md (uppercase) are §-delimited dense prose
        # entries, not markdown bullets — the bullet rule is upstream's lowercase
        # quickstart format. Everything else (length, one paragraph, slop) applies.
        hermes_format = target_path.rsplit("/", 1)[-1] in {"MEMORY.md", "USER.md"}
        if not hermes_format and not text.startswith("-"):
            return "bullet entry required"
        if len(lines) > TARGET_POLICY[target_kind]["max_lines"]:
            return "too many lines for a memory entry"
        if "\n\n" in text:
            return "memory and user entries must stay to one paragraph"
    if target_kind == "skill":
        if not (text.startswith("#") or text.startswith("-")):
            return "skill writebacks must start with a heading or bullet"
        if len(lines) > TARGET_POLICY[target_kind]["max_lines"]:
            return "skill writeback is too long"
    if len(text) > TARGET_POLICY[target_kind]["max_chars"]:
        return "entry exceeds target character budget"
    if len(words) >= 12 and len(set(word.lower() for word in words)) / len(words) < 0.55:
        return "entry looks repetitive or slop-heavy"
    if _GENERIC_SLOP_RE.search(text) and len(words) < 18:
        return "entry is too vague"
    return ""


def _lifecycle_for_fact(parsed: dict[str, Any]) -> str:
    state_value = parsed.get("status") or parsed.get("state") or parsed.get("lifecycle")
    if isinstance(state_value, str):
        normalized = state_value.strip().lower()
        if normalized in {"stale", "superseded", "supersede", "archived"}:
            return normalized
    if isinstance(parsed.get("superseded_by"), str) and str(parsed.get("superseded_by")).strip():
        return "superseded"
    return "active"


def build_idempotence_key(
    *,
    target_kind: str,
    target_path: str,
    mode: str,
    normalized_old_text: str | None,
    normalized_new_text: str | None,
    provenance: Iterable[str],
    policy_version: str = POLICY_VERSION,
) -> str:
    material = "\n".join(
        [
            policy_version,
            target_kind,
            target_path,
            mode,
            normalize_text(normalized_old_text),
            normalize_text(normalized_new_text),
            fingerprint_provenance(provenance),
        ]
    )
    return _sha256(material)[:24]


def stamp_proposal(proposal: DreamProposal, *, policy_version: str = POLICY_VERSION) -> DreamProposal:
    normalized = normalize_text(proposal.proposed_text)
    key = build_idempotence_key(
        target_kind=proposal.target_kind,
        target_path=proposal.target_path,
        mode=proposal.mode,
        normalized_old_text=None,
        normalized_new_text=normalized,
        provenance=proposal.provenance,
        policy_version=policy_version,
    )
    return replace(proposal, proposed_text=normalized, idempotence_key=key, policy_version=policy_version)


def evaluate_proposal(
    proposal: DreamProposal,
    *,
    policy_version: str = POLICY_VERSION,
) -> PolicyDecision:
    warnings: list[str] = []
    target_kind = proposal.target_kind
    if target_kind not in VALID_TARGET_KINDS:
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalize_text(proposal.proposed_text),
            normalized_old_text=None,
            error=f"unsupported target kind {target_kind!r}",
            warnings=warnings,
        )

    normalized_text = normalize_text(proposal.proposed_text)
    normalized_old_text = normalize_text(getattr(proposal, "old_text", None)) or None
    if not _target_path_is_safe(target_kind, proposal.target_path):
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            error=f"unsafe target path {proposal.target_path!r}",
            warnings=warnings,
        )

    if _SECRET_RE.search(proposal.summary) or _SECRET_RE.search(normalized_text):
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            error="proposal contains secret-like content",
            warnings=warnings,
        )

    if not proposal.provenance:
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            error="proposal is missing provenance",
            warnings=warnings,
        )

    if target_kind in {"memory", "user", "skill"} and proposal.mode != "append_text":
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            error=f"{target_kind} proposals must use append_text",
            warnings=warnings,
        )
    if target_kind == "fact" and proposal.mode != "jsonl_append":
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key="",
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            error="fact proposals must use jsonl_append",
            warnings=warnings,
        )

    lifecycle = "active"
    if target_kind == "fact":
        canonical, parsed = _canonical_json_text(normalized_text)
        if canonical is None or parsed is None:
            return PolicyDecision(
                ok=False,
                target_kind=target_kind,
                mode=proposal.mode,
                idempotence_key="",
                policy_version=policy_version,
                normalized_text=normalized_text,
                normalized_old_text=normalized_old_text,
                error="fact proposed_text must be a JSON object string",
                warnings=warnings,
            )
        normalized_text = canonical
        lifecycle = _lifecycle_for_fact(parsed)
        if lifecycle != "active":
            warnings.append(f"fact lifecycle classified as {lifecycle}")
    else:
        entry_problem = _entry_too_sloppy(target_kind, normalized_text, proposal.target_path)
        if entry_problem:
            return PolicyDecision(
                ok=False,
                target_kind=target_kind,
                mode=proposal.mode,
                idempotence_key="",
                policy_version=policy_version,
                normalized_text=normalized_text,
                normalized_old_text=normalized_old_text,
                error=entry_problem,
                warnings=warnings,
            )

    idempotence_key = build_idempotence_key(
        target_kind=target_kind,
        target_path=proposal.target_path,
        mode=proposal.mode,
        normalized_old_text=normalized_old_text,
        normalized_new_text=normalized_text,
        provenance=proposal.provenance,
        policy_version=policy_version,
    )
    capacity = TARGET_POLICY[target_kind]
    if len(normalized_text) > capacity["max_chars"]:
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key=idempotence_key,
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            lifecycle=lifecycle,
            capacity_ok=False,
            capacity_reason=f"entry exceeds {target_kind} char budget",
            error=f"{target_kind} entry exceeds capacity",
            warnings=warnings,
        )
    if normalized_text.count("\n") + 1 > capacity["max_lines"]:
        return PolicyDecision(
            ok=False,
            target_kind=target_kind,
            mode=proposal.mode,
            idempotence_key=idempotence_key,
            policy_version=policy_version,
            normalized_text=normalized_text,
            normalized_old_text=normalized_old_text,
            lifecycle=lifecycle,
            capacity_ok=False,
            capacity_reason=f"entry exceeds {target_kind} line budget",
            error=f"{target_kind} entry exceeds line budget",
            warnings=warnings,
        )

    return PolicyDecision(
        ok=True,
        target_kind=target_kind,
        mode=proposal.mode,
        idempotence_key=idempotence_key,
        policy_version=policy_version,
        normalized_text=normalized_text,
        normalized_old_text=normalized_old_text,
        lifecycle=lifecycle,
        warnings=warnings or None,
    )


def evaluate_live_op(
    *,
    op: str,
    target: str,
    old_text: str | None,
    new_text: str | None,
    reason: str,
    sources: Iterable[str],
    policy_version: str = POLICY_VERSION,
) -> LivePolicyDecision:
    normalized_old = normalize_text(old_text) or None
    normalized_new = normalize_text(new_text) or None
    key = build_idempotence_key(
        target_kind=target,
        target_path=f"{target}.md",
        mode=op,
        normalized_old_text=normalized_old,
        normalized_new_text=normalized_new,
        provenance=sources,
        policy_version=policy_version,
    )
    warnings: list[str] = []
    if target not in VALID_TARGET_KINDS:
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error=f"unsupported target kind {target!r}", warnings=warnings)
    if op not in {"add", "replace", "remove"}:
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error=f"unsupported operation {op!r}", warnings=warnings)
    if not reason.strip():
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error="reason is required", warnings=warnings)
    if not sources or any(not str(source).strip() for source in sources):
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error="sources are required", warnings=warnings)
    if _SECRET_RE.search(reason):
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error="reason contains secret-like content", warnings=warnings)
    if normalized_old and _SECRET_RE.search(normalized_old):
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error="old_text contains secret-like content", warnings=warnings)
    if normalized_new and _SECRET_RE.search(normalized_new):
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error="new_text contains secret-like content", warnings=warnings)

    if op in {"add", "replace"}:
        if not normalized_new:
            return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error=f"{op} requires new_text", warnings=warnings)
        if target in {"memory", "user"} and not normalized_new.startswith("-"):
            return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error=f"{op} new_text must be a bullet entry", warnings=warnings)
    if op in {"replace", "remove"} and not normalized_old:
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, error=f"{op} requires old_text", warnings=warnings)

    lifecycle = "active"
    if target == "fact" and op == "remove":
        lifecycle = "stale"
        warnings.append("facts should be marked stale before removal")

    capacity_reason = ""
    if normalized_new and len(normalized_new) > TARGET_POLICY[target]["max_chars"]:
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, lifecycle=lifecycle, capacity_ok=False, capacity_reason=f"{target} entry exceeds char budget", error=f"{target} entry exceeds capacity", warnings=warnings)
    if normalized_new and normalized_new.count("\n") + 1 > TARGET_POLICY[target]["max_lines"]:
        return LivePolicyDecision(False, target, op, key, policy_version, normalized_old, normalized_new, lifecycle=lifecycle, capacity_ok=False, capacity_reason=f"{target} entry exceeds line budget", error=f"{target} entry exceeds line budget", warnings=warnings)

    return LivePolicyDecision(True, target, op, key, policy_version, normalized_old, normalized_new, lifecycle=lifecycle, warnings=warnings or None)


def run_budget_summary(*, changes: int, adds: int, new_chars: int, targets: int) -> str | None:
    if changes > RUN_POLICY["max_changes"]:
        return f"run exceeds max changes ({changes} > {RUN_POLICY['max_changes']})"
    if adds > RUN_POLICY["max_adds"]:
        return f"run exceeds max adds ({adds} > {RUN_POLICY['max_adds']})"
    if new_chars > RUN_POLICY["max_new_chars"]:
        return f"run exceeds max new chars ({new_chars} > {RUN_POLICY['max_new_chars']})"
    if targets > RUN_POLICY["max_targets"]:
        return f"run exceeds max targets ({targets} > {RUN_POLICY['max_targets']})"
    return None


def policy_thresholds_markdown() -> str:
    return f"""\
| Target | Max chars | Max lines |
|---|---:|---:|
| `memory` | {TARGET_POLICY['memory']['max_chars']} | {TARGET_POLICY['memory']['max_lines']} |
| `user` | {TARGET_POLICY['user']['max_chars']} | {TARGET_POLICY['user']['max_lines']} |
| `skill` | {TARGET_POLICY['skill']['max_chars']} | {TARGET_POLICY['skill']['max_lines']} |
| `fact` | {TARGET_POLICY['fact']['max_chars']} | {TARGET_POLICY['fact']['max_lines']} |

Run budgets:
- max_changes_per_run: {RUN_POLICY['max_changes']}
- max_adds_per_run: {RUN_POLICY['max_adds']}
- max_new_chars_per_run: {RUN_POLICY['max_new_chars']}
- max_targets_per_run: {RUN_POLICY['max_targets']}

Policy version: {POLICY_VERSION}
"""

"""Hermes-native write path for dreaming apply (Tori fork, ported to v0.4.0).

Everything upstream v0.4.0 does NOT know about Hermes' memory files lives here:

- MEMORY.md / USER.md are ``\n§\n``-delimited entry stores with a hard
  per-file injection budget (memory_char_limit / user_char_limit). Appending
  past the budget writes "dark memory" nobody ever sees. The budget gate
  refuses that and DIVERTS the overflow to Cognee (Layer 2) instead.
- Near-duplicate entries (Jaccard >= 0.60 on token sets) are dropped at
  apply time — the bundle builder catches most upstream, this is the net.
- fact-kind proposals are Layer-2 data: they go to Cognee (spooled to disk
  when Cognee is down), with facts.jsonl kept only as a local audit log.
- Verification understands that a dropped-as-duplicate or diverted entry is
  success, not a failed write.

Origin: plugins/hermes-dreaming commit 360b8db (Option-2 re-plumb, 2026-07-14)
+ dreamdedup_2805. Lifted verbatim into its own module on 2026-09-07 so the
upstream ``apply.py`` can track v0.4.0 unchanged; only ``_write_proposal``
delegates here. Constants must track hermes-agent/tools/memory_tool.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .artifact import DreamProposal


def _apply_error(msg: str) -> Exception:
    # One exception type, owned by apply.py, so apply_artifact's rollback
    # `isinstance(exc, DreamApplyError)` sees ours. Late import: apply imports us.
    from .apply import DreamApplyError
    return DreamApplyError(msg)


# Hermes memory_tool.py contract — kept in sync with hermes-agent/tools/
# memory_tool.py::ENTRY_DELIMITER + MemoryStore char_limits + drift guard
# (memory_tool.py::_detect_external_drift). Any change to that file's
# constants requires updating this module.
# ---------------------------------------------------------------------------
HERMES_ENTRY_DELIMITER = "\n§\n"
HERMES_MEMORY_MAX_PER_ENTRY = 2200  # MEMORY.md per-entry char cap (drift guard)
HERMES_USER_MAX_PER_ENTRY = 1375    # USER.md per-entry char cap (drift guard)
HERMES_MEMORY_TARGETS = {"MEMORY.md", "USER.md"}

# Option-2 re-plumb (2026-07-14) — WHOLE-FILE injection budgets. The per-entry
# drift caps above are not the binding constraint: hermes-agent injects at most
# ~memory_char_limit chars of each file into the system prompt. Anything past
# that is written-but-never-seen ("dark memory"). The headroom gate below
# refuses to push a file past its budget; overflow entries divert to Cognee
# (Layer 2 — unbounded, auto-recalled per turn by the memory-router hook).
HERMES_MEMORY_FILE_BUDGET = 2200
HERMES_USER_FILE_BUDGET = 1375
HERMES_FILE_BUDGETS = {
    "MEMORY.md": HERMES_MEMORY_FILE_BUDGET,
    "USER.md": HERMES_USER_FILE_BUDGET,
}

# dreamdedup_2805 (2026-05-28) — application-time near-duplicate gate.
# The bundle builder's existing-state injection (Fix 1) and content-change
# gate (Fix 2) catch most duplicates upstream; this is the belt-and-suspenders
# net at apply time. A proposed memory entry is dropped if its token set is
# >= DEDUP_JACCARD_THRESHOLD similar to ANY existing entry. Jaccard token-set
# overlap (not raw substring) so "Cognee dataset is hermes-tori" and "the
# Cognee default dataset: hermes-tori" collapse, while a genuinely new fact
# that merely shares vocabulary with an existing one survives.
DEDUP_JACCARD_THRESHOLD = 0.60
# Entries shorter than this skip the fuzzy gate (still exact-match deduped) —
# short entries have tiny token sets where Jaccard is noisy and a real new
# fact can spuriously collide with an unrelated short existing entry.
DEDUP_MIN_TOKENS = 6

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercased alphanumeric token set for Jaccard comparison.

    Markdown punctuation, §-noise, and case are stripped so two phrasings of
    the same fact map to near-identical token sets.
    """
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _is_near_duplicate(entry: str, existing_token_sets: list[set[str]]) -> bool:
    """True if ``entry`` is a near-duplicate of any existing entry.

    Short entries (< DEDUP_MIN_TOKENS tokens) are never fuzzy-matched here —
    the caller still applies an exact-match check for those.
    """
    tokens = _tokenize(entry)
    if len(tokens) < DEDUP_MIN_TOKENS:
        return False
    return any(
        _jaccard(tokens, existing) >= DEDUP_JACCARD_THRESHOLD
        for existing in existing_token_sets
        if existing
    )


def _safe_relative_path(path_text: str) -> Path:
    path = PurePosixPath(path_text.replace("\\", "/"))
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise _apply_error(f"unsafe proposal target path: {path_text!r}")
    return Path(*path.parts)


def _resolve_live_path(live_root: Path, proposal: DreamProposal) -> Path:
    return live_root / _safe_relative_path(proposal.target_path)


def _backup_path(backup_root: Path, live_root: Path, target_path: Path) -> Path:
    return backup_root / target_path.relative_to(live_root)


def atomic_write_text(path: Path, text: str) -> None:
    # Single write primitive, owned by apply.py (tests monkeypatch it there to
    # simulate a corrupted write; the Hermes path must be subject to the same).
    from . import apply as _apply
    _apply.atomic_write_text(path, text)

def _apply_append_text(current: str, addition: str) -> str:
    addition = addition.rstrip()
    if addition and addition in current:
        return current if current.endswith("\n") else current + "\n"
    current = current.rstrip()
    if current:
        current += "\n\n"
    current += addition
    if not current.endswith("\n"):
        current += "\n"
    return current


def _split_into_hermes_entries(addition: str, max_per_entry: int) -> list[str]:
    """Split a multi-section markdown blob into Hermes-§-compatible entries.

    Prefers ``## `` markdown headers as section boundaries. Falls back to
    blank-line paragraph boundaries. Each emitted entry MUST be under
    ``max_per_entry`` chars or the drift guard in hermes-agent's
    memory_tool.py will refuse the next memory-tool write.
    """
    addition = addition.strip()
    if not addition:
        return []

    # Cheap path: already small enough → single entry.
    if len(addition) <= max_per_entry:
        return [addition]

    # Try to split on `## ` section headers first.
    parts = re.split(r"\n(?=## )", addition)
    parts = [p.strip() for p in parts if p.strip()]

    # If we still have any oversize part, fall back to paragraph splits inside it.
    expanded: list[str] = []
    for p in parts:
        if len(p) <= max_per_entry:
            expanded.append(p)
            continue
        # Paragraph-level split (blank line separated).
        paras = [pp.strip() for pp in re.split(r"\n\s*\n", p) if pp.strip()]
        if not paras:
            paras = [p]
        # Try to keep the header on its own first entry if possible.
        for para in paras:
            if len(para) <= max_per_entry:
                expanded.append(para)
            else:
                # Hard wrap on max_per_entry boundary (last resort — long
                # contiguous paragraph). Prefer a sentence boundary.
                remaining = para
                while len(remaining) > max_per_entry:
                    cut = remaining.rfind(". ", 0, max_per_entry)
                    if cut < max_per_entry // 2:  # no good break — hard cut
                        cut = max_per_entry
                    else:
                        cut += 1  # include the period
                    expanded.append(remaining[:cut].strip())
                    remaining = remaining[cut:].strip()
                if remaining:
                    expanded.append(remaining)
    return expanded


def _apply_hermes_memory_append(current: str, addition: str, max_per_entry: int) -> str:
    """Append ``addition`` to a Hermes-format memory file (§-separated entries).

    Splits the proposed text into entries that fit Hermes's drift guard,
    deduplicates against existing entries, and joins with ``\\n§\\n``.
    Output round-trips through ``hermes-agent/tools/memory_tool.py`` so the
    next live ``memory(action=add)`` call doesn't trip the drift snapshot.
    """
    existing = [e.strip() for e in current.split(HERMES_ENTRY_DELIMITER) if e.strip()]
    existing_token_sets = [_tokenize(e) for e in existing]
    new_entries = _split_into_hermes_entries(addition, max_per_entry)
    for entry in new_entries:
        # Exact-match dedup (cheap, always on) + Jaccard near-duplicate gate
        # (dreamdedup_2805). Skip entries already covered, verbatim or in
        # substance, by an existing entry.
        if entry in existing:
            continue
        if _is_near_duplicate(entry, existing_token_sets):
            continue
        existing.append(entry)
        existing_token_sets.append(_tokenize(entry))
    body = HERMES_ENTRY_DELIMITER.join(existing)
    if not body.endswith("\n"):
        body += "\n"
    return body


def _apply_hermes_memory_append_budgeted(
    current: str,
    addition: str,
    max_per_entry: int,
    file_budget: int,
) -> tuple[str, list[str], list[str]]:
    """Budget-aware §-append. Returns (updated_text, applied, diverted).

    Same dedup semantics as ``_apply_hermes_memory_append``, plus a
    WHOLE-FILE headroom gate: an entry only lands in the file if the
    resulting file stays within ``file_budget`` chars (the injection
    window). Entries that don't fit are returned in ``diverted`` for the
    caller to route to Cognee — never silently dropped, never written
    into the dark zone past the budget.
    """
    existing = [e.strip() for e in current.split(HERMES_ENTRY_DELIMITER) if e.strip()]
    existing_token_sets = [_tokenize(e) for e in existing]
    new_entries = _split_into_hermes_entries(addition, max_per_entry)
    applied: list[str] = []
    diverted: list[str] = []
    for entry in new_entries:
        if entry in existing:
            continue
        if _is_near_duplicate(entry, existing_token_sets):
            continue
        candidate = existing + [entry]
        candidate_len = len(HERMES_ENTRY_DELIMITER.join(candidate)) + 1  # trailing \n
        if candidate_len > file_budget:
            diverted.append(entry)
            continue
        existing.append(entry)
        existing_token_sets.append(_tokenize(entry))
        applied.append(entry)
    body = HERMES_ENTRY_DELIMITER.join(existing)
    if not body.endswith("\n"):
        body += "\n"
    return body, applied, diverted


def _divert_to_cognee(entries: list[str], *, kind: str, provenance: str = "") -> list[dict]:
    """Route overflow/fact entries to the Layer-2 Cognee brain.

    Import is lazy + guarded so apply never hard-fails on a missing or
    broken sink — cognee_client spools to disk when Cognee is down.
    """
    results: list[dict] = []
    try:
        from .cognee_client import send_fact
    except Exception as exc:  # pragma: no cover — packaging error
        return [{"sent": False, "error": f"cognee_client import failed: {exc}"}]
    for entry in entries:
        results.append(send_fact(entry, kind=kind, provenance=provenance))
    return results


def _apply_hermes_facts_append(current: str, addition: str) -> str:
    """Append a fact (JSONL row or rows) to MEMORY.md as ``[fact] ...`` entries.

    Hermes's memory loader doesn't read ``facts.jsonl``. Routing fact-kind
    proposals to MEMORY.md as plain-text entries (prefixed ``[fact]``) is
    the only way they become visible to the running agent."""
    rows: list[str] = []
    for raw in addition.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Treat as a plain-text fact if not parseable.
            rows.append(f"[fact] {raw}")
            continue
        if isinstance(parsed, dict):
            text = parsed.get("fact") or parsed.get("summary") or json.dumps(parsed, ensure_ascii=False)
            rows.append(f"[fact] {text}")
        else:
            rows.append(f"[fact] {parsed}")

    addition_as_text = HERMES_ENTRY_DELIMITER.join(rows)
    return _apply_hermes_memory_append(current, addition_as_text, HERMES_MEMORY_MAX_PER_ENTRY)


def _apply_jsonl_append(current: str, proposed_text: str) -> str:
    """Append one or more JSON rows to a JSONL file.

    proposed_text may be either a single JSON object, or multiple objects
    one-per-non-empty-line. Each row is normalised (sort_keys, no whitespace
    changes) and deduplicated against existing rows in the file."""
    rows: list[str] = []
    for raw in proposed_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False))
    if not rows:
        rows.append(json.dumps(json.loads(proposed_text.strip()), sort_keys=True, ensure_ascii=False))

    existing = [row.rstrip("\n") for row in current.splitlines() if row.strip()]
    for row in rows:
        if row not in existing:
            existing.append(row)
    return ("\n".join(existing) + "\n") if existing else ""


def _apply_replace_text(proposed_text: str) -> str:
    text = proposed_text.rstrip()
    return text + "\n" if text else ""


def write_hermes_proposal(target: Path, proposal: DreamProposal) -> None:
    target_name = target.name
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    provenance = "; ".join(proposal.provenance) if getattr(proposal, "provenance", None) else ""

    if proposal.mode == "append_text":
        # If we're writing to a Hermes-format memory file, use the
        # budget-aware §-append (Option-2 re-plumb 2026-07-14): entries land
        # in the file only while it stays inside the injection window;
        # overflow diverts to Cognee instead of the dark zone.
        if target_name in HERMES_MEMORY_TARGETS:
            cap = HERMES_USER_MAX_PER_ENTRY if target_name == "USER.md" else HERMES_MEMORY_MAX_PER_ENTRY
            budget = HERMES_FILE_BUDGETS[target_name]
            updated, applied, diverted = _apply_hermes_memory_append_budgeted(
                current, proposal.proposed_text, cap, budget
            )
            if diverted:
                results = _divert_to_cognee(
                    diverted,
                    kind=f"{proposal.target_kind or 'memory'}-overflow",
                    provenance=provenance,
                )
                for entry, res in zip(diverted, results):
                    dest = "cognee" if res.get("sent") else res.get("spooled", "LOST")
                    print(
                        f"[dreaming-apply] {target_name} over budget "
                        f"({budget}c) — diverted {len(entry)}c entry -> {dest}",
                        file=sys.stderr,
                    )
        else:
            updated = _apply_append_text(current, proposal.proposed_text)
    elif proposal.mode == "jsonl_append":
        # Option-2 re-plumb: fact-kind proposals are Layer-2 data. They go
        # to Cognee (unbounded, auto-recalled per turn) — NOT to MEMORY.md,
        # which is reserved for guardrails/IDs under the injection budget.
        # facts.jsonl still gets the raw rows as a local audit log.
        if proposal.target_kind == "fact" and target_name == "facts.jsonl":
            fact_entries: list[str] = []
            for raw in proposal.proposed_text.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        fact_entries.append(
                            str(parsed.get("fact") or parsed.get("summary")
                                or json.dumps(parsed, ensure_ascii=False))
                        )
                    else:
                        fact_entries.append(str(parsed))
                except json.JSONDecodeError:
                    fact_entries.append(raw)
            results = _divert_to_cognee(fact_entries, kind="fact", provenance=provenance)
            sent = sum(1 for r in results if r.get("sent"))
            spooled = sum(1 for r in results if r.get("spooled"))
            print(
                f"[dreaming-apply] fact-kind -> Cognee: {sent} sent, "
                f"{spooled} spooled (of {len(fact_entries)})",
                file=sys.stderr,
            )
            updated = _apply_jsonl_append(current, proposal.proposed_text)
        else:
            updated = _apply_jsonl_append(current, proposal.proposed_text)
    elif proposal.mode == "replace_text":
        updated = _apply_replace_text(proposal.proposed_text)
    else:  # pragma: no cover - guarded by validation
        raise _apply_error(f"unsupported proposal mode: {proposal.mode}")

    atomic_write_text(target, updated)

    verify_text = target.read_text(encoding="utf-8")
    if proposal.mode == "jsonl_append":
        # Every emitted row should be present after the write. Tolerate
        # multi-row payloads (one JSON object per line in proposed_text).
        for raw in proposal.proposed_text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            expected = json.dumps(json.loads(raw), sort_keys=True, ensure_ascii=False)
            if expected not in verify_text:
                raise _apply_error(f"verification failed after writing {target}")
    elif target_name in HERMES_MEMORY_TARGETS:
        # For Hermes-format files, verify by checking the entry round-trips.
        # The original block was split into §-entries; verify any one of
        # them lands. Use the first non-blank line as a unique marker.
        #
        # dreamdedup_2805: the §-append may legitimately DROP an entry as an
        # exact or near-duplicate (its content is already stored). In that
        # case the first line won't appear verbatim — that's success, not
        # failure. Treat "already represented in substance" as verified.
        # Option-2 re-plumb: a budget-DIVERTED entry also won't appear in
        # the file — it went to Cognee/spool instead. That is success too:
        # verify each proposed entry is present, near-duplicate, OR would
        # not fit the file budget (i.e. was legitimately diverted).
        first_line = next(
            (l.strip() for l in proposal.proposed_text.splitlines() if l.strip()),
            None,
        )
        if first_line and first_line not in verify_text:
            existing_after = [
                e.strip() for e in verify_text.split(HERMES_ENTRY_DELIMITER) if e.strip()
            ]
            existing_token_sets = [_tokenize(e) for e in existing_after]
            file_budget = HERMES_FILE_BUDGETS[target_name]
            current_len = len(verify_text)
            proposed_entries = _split_into_hermes_entries(
                proposal.proposed_text,
                HERMES_USER_MAX_PER_ENTRY if target_name == "USER.md" else HERMES_MEMORY_MAX_PER_ENTRY,
            )
            all_accounted = all(
                (entry in existing_after)
                or _is_near_duplicate(entry, existing_token_sets)
                or (current_len + len(HERMES_ENTRY_DELIMITER) + len(entry) > file_budget)
                for entry in proposed_entries
            )
            if not (proposed_entries and all_accounted):
                raise _apply_error(f"verification failed after writing {target}")
    elif proposal.proposed_text.strip() and proposal.proposed_text.strip() not in verify_text:
        raise _apply_error(f"verification failed after writing {target}")

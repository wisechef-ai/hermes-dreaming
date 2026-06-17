"""Tests for dreamdedup_2805 — the near-duplicate gate in _apply_hermes_memory_append.

Run: venv/bin/python3 -m pytest tests/test_dedup.py -q
"""
from __future__ import annotations

from hermes_dreaming.apply import (
    _apply_hermes_memory_append,
    _is_near_duplicate,
    _jaccard,
    _tokenize,
    HERMES_ENTRY_DELIMITER,
    HERMES_MEMORY_MAX_PER_ENTRY,
)


def _entries(blob: str) -> list[str]:
    return [e.strip() for e in blob.split(HERMES_ENTRY_DELIMITER) if e.strip()]


def test_jaccard_basics():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"a", "c"}) == 1 / 3


def test_exact_duplicate_dropped():
    current = "GBrain runs locally on adam-xps with qwen3-embedding"
    out = _apply_hermes_memory_append(current, current, HERMES_MEMORY_MAX_PER_ENTRY)
    assert len(_entries(out)) == 1


def test_near_duplicate_reword_dropped():
    # Same fact, different phrasing — token sets overlap heavily.
    current = "Cognee default dataset is hermes-tori on the sharedbrain host mac01"
    reworded = "The Cognee default dataset hermes-tori lives on sharedbrain host mac01"
    out = _apply_hermes_memory_append(current, reworded, HERMES_MEMORY_MAX_PER_ENTRY)
    assert len(_entries(out)) == 1, "near-duplicate reword should be dropped"


def test_genuinely_new_fact_survives():
    current = "Cognee default dataset is hermes-tori on the sharedbrain host mac01"
    novel = "The carousel-nesting pitfall requires flattening before publish to recipes"
    out = _apply_hermes_memory_append(current, novel, HERMES_MEMORY_MAX_PER_ENTRY)
    assert len(_entries(out)) == 2, "a genuinely new fact must survive"


def test_shared_vocabulary_but_new_fact_survives():
    # Shares some tokens (Cognee, dataset) but is a different fact.
    current = "Cognee default dataset is hermes-tori"
    novel = "Cognee extractor is cognee-extractor:latest running on local Ollama at mac01"
    out = _apply_hermes_memory_append(current, novel, HERMES_MEMORY_MAX_PER_ENTRY)
    assert len(_entries(out)) == 2


def test_short_entry_skips_fuzzy_gate():
    # Below DEDUP_MIN_TOKENS — fuzzy match must not fire (would be noisy).
    assert _is_near_duplicate("auth via x-api-key", [_tokenize("use the x-api-key header")]) is False


def test_empty_current_appends():
    out = _apply_hermes_memory_append("", "A brand new memory fact about the dreaming pipeline efficiency", HERMES_MEMORY_MAX_PER_ENTRY)
    assert len(_entries(out)) == 1


def test_multi_section_blob_dedups_per_section():
    current = "## Alpha\nThe alpha section describes the existing alpha behaviour in detail here"
    addition = (
        "## Alpha\nThe alpha section describes the existing alpha behaviour in detail here\n\n"
        "## Beta\nThe beta section is brand new and describes completely different behaviour now"
    )
    out = _apply_hermes_memory_append(current, addition, HERMES_MEMORY_MAX_PER_ENTRY)
    entries = _entries(out)
    # Alpha is a dup (dropped), Beta is new (kept) → 2 total (orig alpha + new beta).
    assert len(entries) == 2
    assert any("Beta" in e for e in entries)

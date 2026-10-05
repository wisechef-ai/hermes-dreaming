"""Evidence grounding for the anthropic-oauth dreaming provider (2026-10-05).

Regression: the provider's SYSTEM_PROMPT never asked for confidence/reason/
source_quote, so every staged proposal carried confidence=0.0 with empty
evidence (59/59 across 14 nightly artifacts) and the morning reviewer read
the missing score as "unsure" -> DISCARD. These tests pin both halves of the
fix: the prompt asks for the fields, and a missing score is FLAGGED rather
than impersonating a real 0.0 -- including end to end through generate().
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

from hermes_dreaming.artifact import DreamProposal, SourceSnapshot
from hermes_dreaming.providers import DreamContext

mod = importlib.import_module("hermes_dreaming.anthropic_oauth_provider")

BUNDLE = "/home/x/.hermes/dreaming/sources/2026-10-05-vault-recap.md"
CONTENT = "line one\nClawHub search lives at /api/v1/search?q=\nline three\n"


def _src() -> list[SourceSnapshot]:
    return [SourceSnapshot(path=BUNDLE, kind="source", content=CONTENT, sha256="0" * 64,
                           line_count=CONTENT.count("\n"))]


def _raw(**over):
    raw = {
        "id": "fact-x", "target_kind": "fact", "target_path": "facts.jsonl",
        "mode": "jsonl_append", "summary": "s", "provenance": [f"{BUNDLE}:2"],
        "proposed_text": '{"fact":"x"}', "approved": False,
    }
    raw.update(over)
    return raw


def _ground(raws):
    props = [DreamProposal.from_dict(r) for r in raws]
    return mod._ground_evidence(props, raws, _src())


def test_prompt_required_field_list_names_evidence_fields():
    required = next(line for line in mod.SYSTEM_PROMPT.splitlines()
                    if line.startswith("Each proposal must include"))
    for field in ("confidence", "reason", "source_quote"):
        assert field in required, f"required-field list dropped {field}"


def test_prompt_carries_calibration_rubric():
    assert "EVIDENCE FIELDS" in mod.SYSTEM_PROMPT
    assert "0.9+" in mod.SYSTEM_PROMPT and "VERBATIM" in mod.SYSTEM_PROMPT


def test_missing_confidence_is_flagged_not_silently_zero():
    (p,) = _ground([_raw()])
    assert p.confidence == 0.0
    assert mod.CONFIDENCE_UNSCORED_FLAG in p.policy_flags


def test_non_numeric_and_bool_confidence_are_flagged():
    for bad in ("0.9", None, True):
        (p,) = _ground([_raw(confidence=bad)])
        assert mod.CONFIDENCE_UNSCORED_FLAG in p.policy_flags, bad


def test_real_scores_are_kept_and_not_flagged():
    for good in (0.0, 0.85, 1):
        (p,) = _ground([_raw(confidence=good)])
        assert p.confidence == float(good)
        assert mod.CONFIDENCE_UNSCORED_FLAG not in p.policy_flags, good


def test_source_quote_and_snippet_backfilled_from_provenance():
    (p,) = _ground([_raw(confidence=0.9)])
    assert p.source_quote == "ClawHub search lives at /api/v1/search?q="
    assert p.snippet == p.source_quote


def test_basename_provenance_resolves():
    (p,) = _ground([_raw(confidence=0.9, provenance=["2026-10-05-vault-recap.md:2"])])
    assert p.source_quote.startswith("ClawHub")


def test_model_quote_is_never_overwritten():
    (p,) = _ground([_raw(confidence=0.9, source_quote="model quote", snippet="model snip")])
    assert (p.source_quote, p.snippet) == ("model quote", "model snip")


def test_unresolvable_ref_leaves_quote_empty_without_crash():
    for ref in (f"{BUNDLE}:999", f"{BUNDLE}:abc", "no-colon", "/other/file.md:1"):
        (p,) = _ground([_raw(confidence=0.9, provenance=[ref])])
        assert p.source_quote == "", ref


def test_flag_is_idempotent():
    raws = [_raw()]
    props = [DreamProposal.from_dict(r) for r in raws]
    mod._ground_evidence(props, raws, _src())
    mod._ground_evidence(props, raws, _src())
    assert props[0].policy_flags.count(mod.CONFIDENCE_UNSCORED_FLAG) == 1


def test_generate_end_to_end_flags_and_grounds(monkeypatch, tmp_path):
    """The real generate() path must run the grounding step (wiring guard)."""
    payload = {"report": "# r\n- Provider: x", "notes": [], "proposals": [
        _raw(id="unscored"),
        _raw(id="scored", confidence=0.92, reason="durable", source_quote="ClawHub search lives at /api/v1/search?q="),
    ]}

    class _Block:
        type = "text"
        text = "```json\n" + json.dumps(payload) + "\n```"

    class _Messages:
        def create(self, **kwargs):
            assert "source_quote" in kwargs["system"][1]["text"]
            return types.SimpleNamespace(content=[_Block()])

    adapter = types.ModuleType("agent.anthropic_adapter")
    adapter.build_anthropic_client = lambda token: types.SimpleNamespace(messages=_Messages())  # type: ignore[attr-defined]
    creds = types.ModuleType("agent.anthropic_credentials")
    creds.resolve_anthropic_token = lambda: "tok"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.anthropic_adapter", adapter)
    monkeypatch.setitem(sys.modules, "agent.anthropic_credentials", creds)
    monkeypatch.setitem(sys.modules, "anthropic_billing_bypass", None)  # import -> ImportError
    sdk = types.ModuleType("anthropic")  # hermetic: the retry ladder only needs the exception types
    for exc in ("APIStatusError", "APIConnectionError", "APITimeoutError"):
        setattr(sdk, exc, type(exc, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.setattr(mod, "_ensure_hermes_paths", lambda: None)

    ctx = DreamContext(workspace_root=tmp_path, live_root=tmp_path, artifact_dir=tmp_path,
                       source_roots=[Path(BUNDLE).parent])
    _report, proposals, _notes = mod.AnthropicOAuthProvider(model="claude-opus-5-5").generate(_src(), ctx)
    by_id = {p.id: p for p in proposals}
    assert mod.CONFIDENCE_UNSCORED_FLAG in by_id["unscored"].policy_flags
    assert by_id["unscored"].source_quote.startswith("ClawHub")
    assert by_id["scored"].confidence == 0.92
    assert mod.CONFIDENCE_UNSCORED_FLAG not in by_id["scored"].policy_flags

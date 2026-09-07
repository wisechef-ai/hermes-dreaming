from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hermes_dreaming.artifact import SourceSnapshot
from hermes_dreaming.providers import DreamContext, OllamaProvider, OpenAICompatibleProvider, build_provider


class _FakeResponses:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **_kwargs):
        return types.SimpleNamespace(output_text=self._text)


class _FakeOpenAI:
    output_text = ""

    def __init__(self, **_kwargs) -> None:
        self.responses = _FakeResponses(self.output_text)


def _install_fake_openai(monkeypatch, text: str) -> None:
    _FakeOpenAI.output_text = text
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def _context(tmp_path: Path) -> DreamContext:
    return DreamContext(
        workspace_root=tmp_path,
        live_root=tmp_path / "live",
        artifact_dir=tmp_path / "artifact",
        source_roots=[tmp_path / "sources"],
        model="qwen2.5:3b",
    )


def _source() -> SourceSnapshot:
    return SourceSnapshot(
        path="sources/session.md",
        kind="file",
        content="User: Prefer two strong options over six weak ones.",
        sha256="abc123",
        line_count=1,
    )


def test_openai_compatible_provider_accepts_fenced_json_and_forces_unapproved(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openai(
        monkeypatch,
        """```json
{
  "report": "Report body",
  "proposals": [
    {
      "id": "1",
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "User prefers two strong options.",
      "provenance": "session.md:1",
      "confidence": 0.92,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "proposed_text": "- Restaurant design drafts should offer two strong options, not six weak ones.",
      "risk": "medium",
      "priority": "high",
      "reason": "user preference is explicit and actionable",
      "source_quote": "User: Prefer two strong options over six weak ones.",
      "policy_flags": ["profile_preference", "safe_append"],
      "approved": true
    }
  ],
  "notes": "scalar note"
}
```""",
    )

    report, proposals, notes = OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate(
        [_source()], _context(tmp_path)
    )

    assert report == "Report body"
    assert notes == ["scalar note"]
    assert len(proposals) == 1
    assert proposals[0].id == "1"
    assert proposals[0].provenance == ["session.md:1"]
    assert proposals[0].approved is False
    assert proposals[0].confidence == 0.92
    assert proposals[0].snippet == "User: Prefer two strong options over six weak ones."


def test_openai_compatible_provider_rejects_invalid_model_proposals(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openai(
        monkeypatch,
        """{
  "report": "Report body",
  "proposals": [
    {
      "id": "missing-fields",
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "No text should be staged.",
      "provenance": ["session.md:1"],
      "proposed_text": "- Never write this.",
      "approved": true
    },
    {
      "id": "valid",
      "target_kind": "memory",
      "target_path": "memory.md",
      "mode": "append_text",
      "summary": "Keep concise.",
      "provenance": ["session.md:1"],
      "proposed_text": "- Keep concise.",
      "confidence": 0.8,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "approved": false
    }
  ],
  "notes": null
}""",
    )

    with pytest.raises(RuntimeError, match="missing required field"):
        OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate([_source()], _context(tmp_path))


def test_openai_compatible_provider_rejects_fabricated_provenance(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openai(
        monkeypatch,
        """{
  "report": "Report body",
  "proposals": [
    {
      "id": "valid",
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "User prefers two strong options.",
      "provenance": ["made-up:1"],
      "proposed_text": "- Prefer two strong options over six weak ones.",
      "confidence": 0.92,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "risk": "medium",
      "priority": "high",
      "reason": "user preference is explicit and actionable",
      "source_quote": "User: Prefer two strong options over six weak ones.",
      "policy_flags": ["profile_preference", "safe_append"],
      "approved": true
    }
  ],
  "notes": []
}""",
    )

    with pytest.raises(RuntimeError, match="source bundle"):
        OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate([_source()], _context(tmp_path))


def test_openai_compatible_provider_rejects_structured_proposed_text(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openai(
        monkeypatch,
        """{
  "report": "Report body",
  "proposals": [
    {
      "id": "valid",
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "User prefers two strong options.",
      "provenance": ["session.md:1"],
      "proposed_text": {"blob": [1, 2, 3]},
      "confidence": 0.92,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "risk": "medium",
      "priority": "high",
      "reason": "user preference is explicit and actionable",
      "source_quote": "User: Prefer two strong options over six weak ones.",
      "policy_flags": ["profile_preference", "safe_append"],
      "approved": true
    }
  ],
  "notes": []
}""",
    )

    with pytest.raises(RuntimeError, match="proposed_text must be a string"):
        OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate([_source()], _context(tmp_path))


def test_openai_compatible_provider_accepts_bare_numeric_id(monkeypatch, tmp_path: Path) -> None:
    _install_fake_openai(
        monkeypatch,
        """{
  "report": "Report body",
  "proposals": [
    {
      "id": 1,
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "User prefers two strong options.",
      "provenance": "session.md:1",
      "confidence": 0.92,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "proposed_text": "- Restaurant design drafts should offer two strong options, not six weak ones.",
      "risk": "medium",
      "priority": "high",
      "reason": "user preference is explicit and actionable",
      "source_quote": "User: Prefer two strong options over six weak ones.",
      "policy_flags": ["profile_preference", "safe_append"],
      "approved": true
    }
  ],
  "notes": []
}""",
    )

    report, proposals, notes = OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate(
        [_source()], _context(tmp_path)
    )

    assert report == "Report body"
    assert len(proposals) == 1
    assert proposals[0].id == "1"


def test_openai_compatible_provider_splits_comma_joined_provenance(monkeypatch, tmp_path: Path) -> None:
    two_line_source = SourceSnapshot(
        path="sources/session.md",
        kind="file",
        content="User: Prefer two strong options over six weak ones.\nAssistant: Noted.",
        sha256="abc123",
        line_count=2,
    )
    _install_fake_openai(
        monkeypatch,
        """{
  "report": "Report body",
  "proposals": [
    {
      "id": "valid",
      "target_kind": "user",
      "target_path": "user.md",
      "mode": "append_text",
      "summary": "User prefers two strong options.",
      "provenance": "session.md:1,session.md:2",
      "confidence": 0.92,
      "snippet": "User: Prefer two strong options over six weak ones.",
      "proposed_text": "- Restaurant design drafts should offer two strong options, not six weak ones.",
      "risk": "medium",
      "priority": "high",
      "reason": "user preference is explicit and actionable",
      "source_quote": "User: Prefer two strong options over six weak ones.",
      "policy_flags": ["profile_preference", "safe_append"],
      "approved": true
    }
  ],
  "notes": []
}""",
    )

    report, proposals, notes = OpenAICompatibleProvider(model="qwen2.5:3b", api_key="ollama").generate(
        [two_line_source], _context(tmp_path)
    )

    assert len(proposals) == 1
    assert proposals[0].provenance == ["session.md:1", "session.md:2"]


def test_ollama_provider_uses_native_json_chat(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return b'{"message":{"content":"{\\"report\\":\\"Report body\\",\\"proposals\\":[{\\"id\\":\\"p1\\",\\"target_kind\\":\\"user\\",\\"target_path\\":\\"user.md\\",\\"mode\\":\\"append_text\\",\\"summary\\":\\"Keep concise.\\",\\"provenance\\":\\"session.md:1\\",\\"proposed_text\\":\\"- Keep concise.\\",\\"confidence\\":0.92,\\"snippet\\":\\"User: Prefer two strong options over six weak ones.\\",\\"risk\\":\\"medium\\",\\"priority\\":\\"high\\",\\"reason\\":\\"user preference is explicit and actionable\\",\\"source_quote\\":\\"User: Prefer two strong options over six weak ones.\\",\\"policy_flags\\":[\\"profile_preference\\",\\"safe_append\\"],\\"approved\\":true}],\\"notes\\":[]}"}}'

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    report, proposals, notes = OllamaProvider(model="qwen2.5:3b", base_url="http://ollama.test").generate(
        [_source()], _context(tmp_path)
    )

    assert captured["url"] == "http://ollama.test/api/chat"
    assert '"format": "json"' in captured["body"]
    assert '"stream": false' in captured["body"]
    assert report == "Report body"
    assert notes == []
    assert len(proposals) == 1
    assert proposals[0].approved is False
    assert proposals[0].confidence == 0.92
    assert proposals[0].snippet == "User: Prefer two strong options over six weak ones."
    assert proposals[0].provenance == ["session.md:1"]


def test_build_provider_supports_ollama() -> None:
    provider = build_provider("ollama", model="qwen2.5:3b")

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:3b"


def test_list_providers_returns_all_three_with_status() -> None:
    from hermes_dreaming.providers import list_providers

    rows = list_providers()
    names = [row.name for row in rows]
    assert "offline-marker" in names
    assert "openai-compatible" in names
    assert "ollama" in names
    offline = next(row for row in rows if row.name == "offline-marker")
    assert offline.status == "always"
    assert offline.kind == "offline"
    openai_row = next(row for row in rows if row.name == "openai-compatible")
    assert openai_row.status in {"optional", "missing"}
    assert openai_row.kind == "openai_compat"
    ollama_row = next(row for row in rows if row.name == "ollama")
    # We never ping external services; ollama status is "optional" by import-only design.
    assert ollama_row.status == "optional"
    assert ollama_row.kind == "ollama"


def test_render_providers_table_emits_table_with_header_and_separator() -> None:
    from hermes_dreaming.providers import ProviderInfo, list_providers, render_providers_table

    rows = list_providers()
    rendered = render_providers_table(rows)
    assert "NAME" in rendered
    assert "KIND" in rendered
    assert "STATUS" in rendered
    assert "NOTES" in rendered
    assert "----" in rendered
    assert "offline-marker" in rendered
    assert "always" in rendered
    # Sanity: also works for arbitrary rows.
    custom = [ProviderInfo(name="x", kind="k", status="always", notes="n")]
    assert "x" in render_providers_table(custom)


def test_openai_compatible_provider_merges_append_text_proposals_for_same_target() -> None:
    from hermes_dreaming.artifact import DreamProposal
    from hermes_dreaming.providers import OpenAICompatibleProvider

    def mp(pid: str, text: str, prov: str = "src.md:1", confidence: float = 0.9) -> DreamProposal:
        return DreamProposal(
            id=pid, target_kind="memory", target_path="memory.md", mode="append_text",
            summary=f"Note {pid}", provenance=[prov], proposed_text=text,
            approved=False, confidence=confidence, snippet="", risk="low",
            priority="normal", reason="test", source_quote="", policy_flags=["test"],
        )

    provider = OpenAICompatibleProvider(api_key="test")
    proposals = [
        mp("p1", "- Keep notes short.", "src1.md:1", 0.9),
        mp("p2", "- Use concrete examples.", "src2.md:1", 0.8),
    ]
    result = provider._dedupe_proposals(proposals, payload_hash="test")
    assert len(result) == 1
    assert "- Keep notes short." in result[0].proposed_text
    assert "- Use concrete examples." in result[0].proposed_text
    assert len(result[0].provenance) == 2


def test_openai_compatible_provider_dedupes_identical_append_text() -> None:
    from hermes_dreaming.artifact import DreamProposal
    from hermes_dreaming.providers import OpenAICompatibleProvider

    def mp(pid: str, text: str, prov: str = "src.md:1") -> DreamProposal:
        return DreamProposal(
            id=pid, target_kind="memory", target_path="memory.md", mode="append_text",
            summary=f"Note {pid}", provenance=[prov], proposed_text=text,
            approved=False, snippet="", risk="low", priority="normal",
            reason="test", source_quote="", policy_flags=["test"],
        )

    provider = OpenAICompatibleProvider(api_key="test")
    dup = [
        mp("p3", "- Same note.", "src1.md:1"),
        mp("p4", "- Same note.", "src2.md:1"),
    ]
    result = provider._dedupe_proposals(dup, payload_hash="test")
    assert len(result) == 1
    assert result[0].proposed_text == "- Same note."
    assert len(result[0].provenance) == 2


def test_openai_compatible_provider_keeps_distinct_jsonl_rows_and_collapses_identical() -> None:
    """Distinct jsonl_append rows to one file are additive (one JSON object per
    row is the whole point of the mode); byte-identical rows collapse to one.

    Tori fork divergence (2026-09-07): upstream errored here, which turned every
    multi-fact night into an `invalid` artifact on a live model run. Proposed
    upstream; keep this pin until it lands.
    """
    from hermes_dreaming.artifact import DreamProposal
    from hermes_dreaming.providers import OpenAICompatibleProvider

    def mp(pid: str, text: str, prov: str = "src.md:1") -> DreamProposal:
        return DreamProposal(
            id=pid, target_kind="fact", target_path="facts.jsonl", mode="jsonl_append",
            summary=f"Fact {pid}", provenance=[prov], proposed_text=text,
            approved=False, snippet="", risk="low", priority="normal",
            reason="test", source_quote="", policy_flags=["test"],
        )

    provider = OpenAICompatibleProvider(api_key="test")
    out = provider._dedupe_proposals(
        [mp("j1", '{"key":"a"}', "src1.md:1"), mp("j2", '{"key":"b"}', "src2.md:1"), mp("j3", '{"key":"a"}', "src3.md:1")],
        payload_hash="test",
    )
    assert sorted(p.proposed_text for p in out) == ['{"key":"a"}', '{"key":"b"}']
    dup = next(p for p in out if p.proposed_text == '{"key":"a"}')
    assert set(dup.provenance) == {"src1.md:1", "src3.md:1"}


def test_openai_compatible_provider_drops_only_the_policy_invalid_proposal() -> None:
    """One bad proposal must not void the batch (fail-partial, not fail-closed).

    Tori fork divergence (2026-09-07): a live run produced 9 proposals; one
    memory bullet was 5 lines (policy max 4) and the whole artifact went
    `invalid` with 0 proposals. The 8 good ones are kept, the bad one is
    reported as a note.
    """
    from hermes_dreaming.artifact import DreamProposal
    from hermes_dreaming.providers import OpenAICompatibleProvider

    def mp(pid: str, text: str) -> DreamProposal:
        return DreamProposal(
            id=pid, target_kind="memory", target_path="memory.md", mode="append_text",
            summary=f"Memory {pid}", provenance=["src.md:1"], proposed_text=text,
            approved=False, snippet="", risk="low", priority="normal",
            reason="test", source_quote="", policy_flags=["test"],
        )

    good = mp("good", "- one concise bullet")
    bad = mp("bad", "- line1\n- line2\n- line3\n- line4\n- line5")
    kept, dropped = OpenAICompatibleProvider._drop_policy_invalid([good, bad])
    assert [p.id for p in kept] == ["good"]
    assert dropped and dropped[0][0] == "bad" and "too many lines" in dropped[0][1]

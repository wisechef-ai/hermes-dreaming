"""Anthropic OAuth provider for hermes-dreaming.

Reuses the running Hermes venv's anthropic_adapter (OAuth token resolution,
client construction) and the Claude Code OAuth bypass patches that the
sitecustomize hook already injects. This lets `hermes dreaming -- create
--provider anthropic --model opus` route through the Claude Max subscription
without paying API tokens.

Design notes
------------
- We don't reimplement billing-header signing, mcp->x rename, or httpx
  header rewriting. The Hermes patches package (`~/.hermes/patches/
  anthropic_billing_bypass.py`) does all of that as soon as the Hermes
  venv's site-packages sitecustomize.py runs at interpreter start.
- We call `apply_claude_code_bypass(kwargs, version)` ourselves on the
  outbound kwargs before `client.messages.create(**kwargs)`. The httpx
  send hook is process-global and applies automatically.
- Token resolution prefers `~/.claude/.credentials.json` via
  `agent.anthropic_adapter.resolve_anthropic_token()`, which also handles
  expiry-aware refresh.
- This is NOT for raw API-key Anthropic. If you ever provision a real
  ANTHROPIC_API_KEY, set it in env and resolve_anthropic_token() will
  pick it up — the bypass shape happens to be a no-op for API keys.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact import DreamProposal, SourceSnapshot
from .providers import DreamContext, MARKER_RE


# Map convenience aliases -> real Anthropic model IDs available on Claude Max.
# Kept in sync with hermes-claude-oauth-bypass skill (Available Models table).
_MODEL_ALIASES = {
    "opus":   "claude-opus-4-7",
    "opus-4": "claude-opus-4-7",
    "opus-4-6": "claude-opus-4-6",
    "opus-4-7": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4-5": "claude-sonnet-4-5",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku":  "claude-haiku-4-5",
    "haiku-4-5": "claude-haiku-4-5",
}

DEFAULT_MODEL = "claude-opus-4-7"


def _ensure_hermes_paths() -> None:
    """Make the Hermes agent + patches importable when the dreaming CLI is
    launched from the Hermes venv. Idempotent."""
    candidates = [
        Path.home() / ".hermes" / "hermes-agent",
        Path.home() / ".hermes" / "patches",
    ]
    for p in candidates:
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def _resolve_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    key = model.strip().lower()
    return _MODEL_ALIASES.get(key, model)


def _get_cli_version() -> str:
    """Fallback CLI version string; the bypass module owns the canonical
    detection but exposes it indirectly. Hardcoding a recent default is fine
    because the bypass overrides it on the wire via httpx hook."""
    return os.environ.get("HERMES_DREAMING_CLI_VERSION", "2.1.117")


SYSTEM_PROMPT = (
    "You are Hermes Dreaming, a staged self-improvement engine for the Hermes\n"
    "agent. Return JSON only with keys: report, proposals, notes.\n"
    "Each proposal must include id, target_kind, target_path, mode, summary, "
    "provenance, proposed_text, approved.\n"
    "\n"
    "STRICT FIELD CONTRACT:\n"
    "  - target_kind in {memory, user, skill, fact}\n"
    "  - mode in {append_text, jsonl_append}\n"
    "  - target_path is a RELATIVE path under live_root with NO leading slash.\n"
    "    * memory updates: target_path = \"MEMORY.md\"\n"
    "    * user-profile updates: target_path = \"USER.md\"\n"
    "    * fact appends: target_path = \"facts.jsonl\" (mode jsonl_append)\n"
    "    * skill updates: target_path = \"skills/<category>/<name>/SKILL.md\"\n"
    "  - For mode=jsonl_append, proposed_text may contain EITHER one JSON\n"
    "    object OR multiple JSON objects one-per-line (true JSONL). When\n"
    "    you have several distinct facts to record, prefer ONE jsonl_append\n"
    "    proposal whose proposed_text is multi-line JSONL — that keeps the\n"
    "    artifact compact and avoids same-target proposal sprawl.\n"
    "  - provenance MUST be a JSON array of STRINGS, where each string is a\n"
    "    source pointer like \"<absolute-or-relative-path>:<line-number>\".\n"
    "    Do NOT split a path into one character per array element.\n"
    "  - approved MUST be false (humans approve later).\n"
    "  - id is a short kebab-case slug; keep it stable across re-runs of the\n"
    "    same source content.\n"
    "\n"
    "HERMES MEMORY FILE CONTRACT (for target_kind in {memory, user}):\n"
    "  - MEMORY.md and USER.md are read by hermes-agent's memory_tool.py.\n"
    "  - Each ENTRY in those files is a self-contained, atomic, recall-keyed\n"
    "    fact or rule, joined to its neighbours by `\\n§\\n` (section sign).\n"
    "  - Per-entry hard limit: 2200 chars for MEMORY.md, 1375 chars for\n"
    "    USER.md. Anything over trips the drift guard and breaks future\n"
    "    memory writes.\n"
    "  - YOUR proposed_text should be AT MOST ~400 chars per atomic entry.\n"
    "    If you have multiple distinct ideas to capture, emit MULTIPLE\n"
    "    proposals (one per atomic entry), NOT one big proposal with several\n"
    "    `##` headings. Smaller entries recall better and survive the\n"
    "    char-budget pressure.\n"
    "  - Write entries in the dense, abbreviated style of MEMORY.md (codename:\n"
    "    detail; PREF:/RULE: prefixes; ID + value pairs; no narrative prose).\n"
    "    Look at the source bundle for examples of the house style.\n"
    "\n"
    "FACT ROUTING (for target_kind=fact):\n"
    "  - Hermes does NOT have a facts.jsonl loader. The apply step routes\n"
    "    fact-kind proposals to MEMORY.md as `[fact] ...` §-entries automatically.\n"
    "  - You can still emit JSONL — the apply step pulls the `fact` or\n"
    "    `summary` field and writes `[fact] <text>` as one memory entry per row.\n"
    "  - Keep each row's `fact`/`summary` field short and self-contained,\n"
    "    under 400 chars, written in the dense MEMORY.md house style.\n"
    "\n"
    "Never include secrets, tokens, or hardcoded personal data.\n"
    "\n"
    "JSON OUTPUT RULES (CRITICAL):\n"
    "  - Newlines inside any string value MUST be escaped as \\\\n.\n"
    "  - Tabs as \\\\t. Double quotes inside strings as \\\\\". Backslashes as \\\\\\\\.\n"
    "  - Do NOT emit raw control characters inside strings.\n"
    "  - No trailing commas. Wrap output in a single fenced ```json ... ``` block.\n"
    "\n"
    "When the source bundle contains explicit `DREAM:` markers, treat them "
    "as ground truth and emit one proposal per marker. When the bundle is "
    "unstructured prose, distill the most useful, low-risk, atomic memory/"
    "user/skill/fact updates — prefer fewer high-signal proposals to many "
    "low-signal ones."
)


def _build_user_prompt(sources: list[SourceSnapshot], context: DreamContext) -> str:
    blocks = []
    for source in sources:
        blocks.append(f"### {source.path}\n{source.content}")
    body = "\n\n".join(blocks)
    return (
        f"Workspace root: {context.workspace_root}\n"
        f"Live root: {context.live_root}\n"
        f"Source roots: {', '.join(str(p) for p in context.source_roots)}\n"
        f"\nSources:\n{body}\n"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly parse the model's JSON response.

    Handles three common failure modes:
      1. Fenced output: ```json ... ``` or ``` ... ```
      2. Stray prose before/after the JSON object
      3. Trailing commas (json.loads rejects, json5 not always available)
      4. Unescaped control chars inside strings (newlines, tabs in proposed_text)

    On parse failure, retries with progressively more permissive cleanup."""
    import re as _re

    text = text.strip()
    # Strip code fences.
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rstrip("`").rstrip()

    # Narrow to the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    # First pass: vanilla parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second pass: strip trailing commas before } or ].
    cleaned = _re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Third pass: strict=False permits literal newlines / tabs / control chars
    # inside JSON strings. Opus has been observed embedding raw \n inside
    # `proposed_text` instead of `\\n` when it gets long.
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Fourth pass: Opus truncates mid-JSON at max_tokens boundary, leaving
    # unterminated strings at the end (e.g. "text": "Lorem ipsum...truncated).
    # Walk backwards to find the last unmatched quote, close it, and close
    # all open containers ({ [ that remain.
    try:
        stack = []  # track open braces/brackets
        in_string = False
        escape_next = False
        last_quote_pos = -1

        # Forward pass: track bracket/brace depth and string state
        for i in range(len(cleaned)):
            c = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue

            if c in ('"', "'") and not in_string:
                in_string = True
                last_quote_pos = i
            elif c in ('"', "'") and in_string:
                in_string = False
            elif not in_string:
                if c in "{[":
                    stack.append(c)
                elif c in "}]":
                    if stack:
                        stack.pop()

        # If a string is still open, close it and any remaining containers
        if in_string and last_quote_pos >= 0:
            # Close the unterminated string with its matching quote
            quote_char = cleaned[last_quote_pos]
            fixed = cleaned + quote_char
            # Add closing braces/brackets for any still-open containers
            for bracket in reversed(stack):
                if bracket == "{":
                    fixed += "}"
                else:
                    fixed += "]"
            cleaned = fixed

        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"unparseable after 4 cleanup passes: {exc.msg}", exc.doc, exc.pos
        )


@dataclass(slots=True)
class AnthropicOAuthProvider:
    """Claude Max-backed dreaming provider.

    Builds proposals from source snapshots by calling the Anthropic Messages
    API through the OAuth bypass infrastructure already installed in the
    Hermes venv. If the source bundle contains explicit `DREAM:` markers,
    those are surfaced verbatim to the model as ground truth.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 16384
    name: str = "anthropic-oauth"

    def __post_init__(self) -> None:
        self.model = _resolve_model(self.model)

    def generate(
        self,
        sources: list[SourceSnapshot],
        context: DreamContext,
    ) -> tuple[str, list[DreamProposal], list[str]]:
        _ensure_hermes_paths()
        try:
            from agent.anthropic_adapter import (
                resolve_anthropic_token,
                build_anthropic_client,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hermes agent.anthropic_adapter is not importable. "
                "The anthropic provider requires the Hermes venv to be active "
                f"(running interpreter: {sys.executable})."
            ) from exc

        token = resolve_anthropic_token()
        if not token:
            raise RuntimeError(
                "No Anthropic token resolved. Expected Claude Max OAuth in "
                "~/.claude/.credentials.json or ANTHROPIC_API_KEY in env."
            )

        # Try to import the bypass; if absent, we still proceed (raw API key
        # path) but warn loudly.
        bypass_apply = None
        try:
            from anthropic_billing_bypass import apply_claude_code_bypass
            bypass_apply = apply_claude_code_bypass
        except ImportError:
            pass

        client = build_anthropic_client(token)

        # Build the raw kwargs in a shape compatible with the OAuth bypass.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                },
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": _build_user_prompt(sources, context),
                }
            ],
        }

        # Strip temperature on Opus 4.6/4.7 to avoid the adaptive-thinking 400.
        # (The bypass also handles this on the main adapter path, but we're
        # calling messages.create directly so we own it here.)
        is_opus = "4-6" in self.model or "4-7" in self.model or "4.6" in self.model or "4.7" in self.model
        if not is_opus:
            kwargs["temperature"] = 0

        if bypass_apply is not None:
            try:
                bypass_apply(kwargs, _get_cli_version())
            except Exception as exc:  # pragma: no cover — defensive
                # The bypass is best-effort. If signing fails we still try
                # the request (which may 400 — caller sees the real error).
                print(
                    f"[anthropic-oauth-provider] bypass apply failed: {exc!r}",
                    file=sys.stderr,
                )

        # Retry ladder for transient API failures (added 2026-07-14 after a
        # naked 529 Overloaded killed the 03:45 nightly run with zero retry).
        # 529/429/5xx are transient by definition — a years-scale pipeline
        # never makes a single naked network call. 4 attempts, exponential
        # backoff 30s/60s/120s (nightly cron has no latency budget).
        import anthropic as _anthropic
        import time as _time

        _RETRYABLE = (
            _anthropic.APIStatusError,
            _anthropic.APIConnectionError,
            _anthropic.APITimeoutError,
        )
        max_attempts = 4
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.messages.create(**kwargs)
                break
            except _RETRYABLE as exc:
                status = getattr(exc, "status_code", None)
                transient = status is None or status in (429, 500, 502, 503, 529)
                if not transient or attempt == max_attempts:
                    raise
                delay = 30 * (2 ** (attempt - 1))
                print(
                    f"[anthropic-oauth-provider] transient API error "
                    f"(status={status}, attempt {attempt}/{max_attempts}) — "
                    f"retrying in {delay}s: {exc!r}",
                    file=sys.stderr,
                )
                _time.sleep(delay)
        assert response is not None  # loop either broke or raised

        # Pull the text out of the Anthropic response.
        text_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()
        if not text:
            raise RuntimeError("Anthropic returned no text content")

        try:
            payload = _extract_json(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Anthropic returned non-JSON output: {exc}\n"
                f"--- raw output ---\n{text[:2000]}"
            ) from exc

        report = str(
            payload.get(
                "report",
                f"# Hermes Dreaming Report\n\n- Provider: `{self.name}`\n- Model: `{self.model}`\n",
            )
        )
        proposals_raw = payload.get("proposals", []) or []
        proposals = [DreamProposal.from_dict(item) for item in proposals_raw]
        proposals = _normalize_provenance(proposals, sources)
        notes = [str(item) for item in payload.get("notes", []) or []]

        # True fallback ONLY: if the model returned zero proposals but the
        # source bundle contained explicit DREAM: markers, harvest them so
        # the offline-marker contract still holds. When the model returned
        # any proposals at all we trust its judgement — it saw the markers
        # in the prompt.
        if not proposals:
            proposals.extend(_harvest_markers(sources))

        # Coalesce same-target append_text proposals into one (the validator
        # rejects multiple distinct append_text bodies against the same file).
        proposals = _coalesce_appends(proposals)

        proposals.sort(key=lambda item: (item.target_kind, item.target_path, item.id))

        # Stamp the provider name into the report header if the model omitted it.
        if "Provider" not in report.splitlines()[0:5] and "provider" not in report.lower()[:200]:
            report = (
                f"# Hermes Dreaming Report\n\n"
                f"- Provider: `{self.name}`\n"
                f"- Model: `{self.model}`\n"
                f"- Sources scanned: `{len(sources)}`\n"
                f"- Proposals staged: `{len(proposals)}`\n\n"
                f"{report}"
            )

        return report, proposals, notes


def _normalize_provenance(
    proposals: list[DreamProposal],
    sources: list[SourceSnapshot],
) -> list[DreamProposal]:
    """Fix up provenance fields that the model mangled.

    Opus has been observed splitting a string path into one-char-per-element
    arrays. Detect that shape and rejoin. Also default to the source paths
    when provenance is empty or all-empty-strings."""
    source_paths = [s.path for s in sources]
    out: list[DreamProposal] = []
    for p in proposals:
        prov = list(p.provenance or [])
        # If every element is a single character, the model wrote a string
        # where it should have written one entry. Rejoin.
        if prov and all(isinstance(x, str) and len(x) == 1 for x in prov):
            joined = "".join(prov).strip()
            prov = [joined] if joined else []
        # Empty / placeholder provenance → fall back to source list.
        if not prov or all((not str(x).strip()) for x in prov):
            prov = list(source_paths) or ["<unknown>"]
        p.provenance = prov
        out.append(p)
    return out


def _coalesce_appends(proposals: list[DreamProposal]) -> list[DreamProposal]:
    """Merge same-target append_text proposals into a single proposal.

    The validator rejects multiple distinct append_text bodies against the
    same file as a conflict. When the LLM emits two memory updates that
    both land in MEMORY.md, that's not a conflict — it's two things to
    append. Concatenate their bodies, merge provenance, keep the first id.
    Non-append_text proposals pass through unchanged."""
    grouped: dict[tuple[str, str, str], list[DreamProposal]] = {}
    order: list[tuple[str, str, str]] = []
    passthrough: list[DreamProposal] = []
    for p in proposals:
        if p.mode != "append_text":
            passthrough.append(p)
            continue
        key = (p.target_kind, p.target_path, p.mode)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(p)

    merged: list[DreamProposal] = []
    for key in order:
        group = grouped[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        first = group[0]
        bodies: list[str] = []
        provenance: list[str] = []
        summaries: list[str] = []
        for item in group:
            text = item.proposed_text.rstrip("\n")
            if text and text not in bodies:
                bodies.append(text)
            for prov in item.provenance:
                if prov not in provenance:
                    provenance.append(prov)
            if item.summary and item.summary not in summaries:
                summaries.append(item.summary)
        first.proposed_text = "\n".join(bodies) + "\n"
        first.provenance = provenance
        first.summary = " | ".join(summaries)[:300]
        first.notes = (first.notes or "") + (
            f" [coalesced from {len(group)} proposals: " + ", ".join(g.id for g in group) + "]"
        )
        merged.append(first)

    return merged + passthrough


def _harvest_markers(sources: list[SourceSnapshot]) -> list[DreamProposal]:
    """Fallback: scan for DREAM: markers and stage them as proposals even if
    the LLM forgot. Mirrors OfflineMarkerProvider's parsing."""
    from .providers import OfflineMarkerProvider

    offline = OfflineMarkerProvider()
    out: list[DreamProposal] = []
    for source in sources:
        for line_number, line in enumerate(source.content.splitlines(), start=1):
            match = MARKER_RE.match(line)
            if not match:
                continue
            kind, payload = match.groups()
            proposal = offline._build_proposal(kind.lower(), payload.strip(), source, line_number)
            if proposal is not None:
                out.append(proposal)
    return out

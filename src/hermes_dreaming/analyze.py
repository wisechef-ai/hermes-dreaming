from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import sys
import re
from pathlib import Path
from uuid import uuid4

from .artifact import DreamArtifact, write_artifact
from .collect import collect_sources
from .commands.harvest import build_recent_source_snapshot
from .policy import stamp_proposal
from .providers import DreamContext, build_provider
from .triage import ProposalView, aggregate_policy_flags, proposal_view, sorted_proposals
from .validation import validate_artifact, validate_source_snapshots


@dataclass(slots=True)
class DreamCreationResult:
    artifact: DreamArtifact
    artifact_dir: Path
    validation_errors: list[str]


@dataclass(slots=True)
class DreamRunConfig:
    live_root: Path
    artifact_root: Path
    source_paths: list[Path]
    recent_limit: int | None = None
    provider_name: str = "offline-marker"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True)
class DreamReportCard:
    artifact_id: str
    created_at: str
    provider: str
    status: str
    source_count: int
    proposal_count: int
    target_kind_breakdown: dict[str, int]
    validation_state: str
    validation_error_count: int
    validation_errors: list[str]
    apply_state: str
    discard_state: str
    theme_labels: list[str]
    applied_proposal_ids: list[str]
    backup_count: int
    policy_flags: list[str]
    proposal_views: list[ProposalView]


def _redact_shareable_text(text: str) -> str:
    return re.sub(r"(?i)\b[\w-]*secret[\w-]*\b", "[REDACTED]", text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_artifact_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def build_report(artifact: DreamArtifact) -> str:
    lines = [
        "# Hermes Dreaming Report",
        "",
        f"- Artifact: `{artifact.artifact_id}`",
        f"- Created: `{artifact.created_at}`",
        f"- Provider: `{artifact.provider}`",
        f"- Status: `{artifact.status}`",
        f"- Sources scanned: `{len(artifact.sources)}`",
        f"- Proposals staged: `{len(artifact.proposals)}`",
        "",
    ]
    if artifact.validation_errors:
        lines.extend(["## Validation", ""])
        for error in artifact.validation_errors:
            lines.append(f"- {error}")
        lines.append("")
    lines.extend(["## Proposals", ""])
    if artifact.proposals:
        for proposal in artifact.proposals:
            lines.append(f"- `{proposal.id}` -> `{proposal.target_path}` ({proposal.mode})")
            lines.append(f"  - {proposal.summary}")
            lines.append(f"  - Confidence: `{proposal.confidence:.2f}`")
            lines.append(f"  - Risk/Priority: `{proposal.risk}` / `{proposal.priority}`")
            if proposal.reason:
                lines.append(f"  - Reason: {proposal.reason}")
            if proposal.source_quote:
                lines.append(f"  - Source quote: {proposal.source_quote}")
            if proposal.policy_flags:
                lines.append(f"  - Policy flags: {', '.join(proposal.policy_flags)}")
            lines.append(f"  - Snippet: {proposal.snippet}")
            lines.append(f"  - Provenance: {', '.join(proposal.provenance)}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _build_failure_report(
    artifact_id: str,
    *,
    provider_name: str,
    error_message: str,
    source_count: int,
    payload_hash: str | None = None,
) -> str:
    lines = [
        "# Hermes Dreaming Report",
        "",
        f"- Artifact: `{artifact_id}`",
        f"- Provider: `{provider_name}`",
        "- Status: `invalid`",
        f"- Sources scanned: `{source_count}`",
        "- Proposals staged: `0`",
        "",
        "## Provider/preflight failure",
        "",
        f"- Error: {error_message}",
    ]
    if payload_hash:
        lines.append(f"- Payload hash: `{payload_hash}`")
    lines.append("")
    return "\n".join(lines)


def _theme_label_for_target_kind(target_kind: str) -> str:
    return {
        "memory": "memory updates",
        "user": "user updates",
        "skill": "skill updates",
        "fact": "fact updates",
    }.get(target_kind, f"{target_kind} updates")


def build_report_card(artifact: DreamArtifact) -> DreamReportCard:
    target_kind_breakdown = dict(sorted(Counter(proposal.target_kind for proposal in artifact.proposals).items()))
    theme_labels = [_theme_label_for_target_kind(target_kind) for target_kind in target_kind_breakdown]
    validation_errors = list(artifact.validation_errors)
    validation_state = "invalid" if validation_errors or artifact.status == "invalid" else "valid"
    apply_state = "applied" if artifact.applied_at or artifact.status == "applied" else "not applied"
    discard_state = "discarded" if artifact.discarded_at or artifact.status == "discarded" else "not discarded"
    proposal_views = []
    for proposal in sorted_proposals(artifact.proposals):
        view = proposal_view(proposal)
        proposal_views.append(
            ProposalView(
                id=view.id,
                state=view.state,
                target_kind=view.target_kind,
                target_path=view.target_path,
                summary=_redact_shareable_text(view.summary),
                confidence=view.confidence,
                risk=view.risk,
                priority=view.priority,
                reason=_redact_shareable_text(view.reason),
                source_quote=_redact_shareable_text(view.source_quote),
                policy_flags=list(view.policy_flags),
                provenance=list(view.provenance),
            )
        )

    return DreamReportCard(
        artifact_id=artifact.artifact_id,
        created_at=artifact.created_at,
        provider=artifact.provider,
        status=artifact.status,
        source_count=len(artifact.sources),
        proposal_count=len(artifact.proposals),
        target_kind_breakdown=target_kind_breakdown,
        validation_state=validation_state,
        validation_error_count=len(validation_errors),
        validation_errors=validation_errors,
        apply_state=apply_state,
        discard_state=discard_state,
        theme_labels=theme_labels,
        applied_proposal_ids=list(artifact.applied_proposal_ids),
        backup_count=len(artifact.backup_paths),
        policy_flags=aggregate_policy_flags(artifact.proposals),
        proposal_views=proposal_views,
    )


def render_report_card_markdown(card: DreamReportCard) -> str:
    lines = [
        "# Hermes Dreaming report card",
        "",
        f"- Artifact: `{card.artifact_id}`",
        f"- Created: `{card.created_at}`",
        f"- Provider: `{card.provider}`",
        f"- Status: `{card.status}`",
        f"- Sources scanned: `{card.source_count}`",
        f"- Proposals staged: `{card.proposal_count}`",
        f"- Validation state: `{card.validation_state}`",
        f"- Apply state: `{card.apply_state}`",
        f"- Discard state: `{card.discard_state}`",
        "",
        "## Target kinds",
        "",
    ]
    if card.target_kind_breakdown:
        for target_kind, count in card.target_kind_breakdown.items():
            lines.append(f"- `{target_kind}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Theme labels",
        "",
    ])
    if card.theme_labels:
        for label in card.theme_labels:
            lines.append(f"- {label}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Proposal triage",
        "",
        f"- Policy flags: {', '.join(card.policy_flags) if card.policy_flags else 'none'}",
    ])
    if card.proposal_views:
        for proposal in card.proposal_views:
            lines.append(
                f"- `{proposal.id}` [{proposal.state}] `{proposal.target_kind}` -> `{proposal.target_path}`"
            )
            lines.append(f"  - summary: {proposal.summary}")
            lines.append(f"  - risk/priority: `{proposal.risk}` / `{proposal.priority}`")
            lines.append(f"  - confidence: `{proposal.confidence:.2f}`")
            if proposal.reason:
                lines.append(f"  - reason: {proposal.reason}")
            if proposal.source_quote:
                lines.append(f"  - source quote: {proposal.source_quote}")
            if proposal.policy_flags:
                lines.append(f"  - policy flags: {', '.join(proposal.policy_flags)}")
            if proposal.provenance:
                lines.append(f"  - provenance: {', '.join(proposal.provenance)}")
            else:
                lines.append("  - provenance: none")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Apply details",
        "",
        f"- Applied proposal ids: {', '.join(f'`{proposal_id}`' for proposal_id in card.applied_proposal_ids) if card.applied_proposal_ids else 'none'}",
        f"- Backup copies: `{card.backup_count}`",
        "",
        "## Validation",
        "",
        f"- Error count: `{card.validation_error_count}`",
    ])
    if card.validation_errors:
        for error in card.validation_errors:
            lines.append(f"- {error}")
    else:
        lines.append("- none")

    return "\n".join(lines).rstrip() + "\n"


def render_report_card_json(card: DreamReportCard) -> str:
    return json.dumps(asdict(card), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def create_dream_artifact(config: DreamRunConfig) -> DreamCreationResult:
    live_root = Path(config.live_root)
    artifact_root = Path(config.artifact_root)
    source_paths = [Path(path) for path in config.source_paths]

    source_snapshots = collect_sources(source_paths)
    if config.recent_limit is not None:
        recent_snapshot, _sessions, _redactions = build_recent_source_snapshot(recent=config.recent_limit)
        source_snapshots.append(recent_snapshot)
    artifact_id = generate_artifact_id()
    artifact_dir = artifact_root / artifact_id

    provider = build_provider(config.provider_name, model=config.model, api_key=config.api_key, base_url=config.base_url)
    source_validation_errors = validate_source_snapshots(source_snapshots)
    if source_validation_errors:
        report_body = _build_failure_report(
            artifact_id,
            provider_name=provider.name,
            error_message="source preflight blocked provider call: " + "; ".join(source_validation_errors),
            source_count=len(source_snapshots),
        )
        artifact = DreamArtifact(
            artifact_id=artifact_id,
            created_at=_now_iso(),
            provider=provider.name,
            status="invalid",
            workspace_root=str(live_root),
            source_roots=[str(path) for path in source_paths],
            report=report_body.rstrip() + "\n",
            sources=[],
            proposals=[],
            validation_errors=source_validation_errors,
        )
        write_artifact(artifact, artifact_dir)
        return DreamCreationResult(
            artifact=artifact,
            artifact_dir=artifact_dir,
            validation_errors=source_validation_errors,
        )

    context = DreamContext(
        workspace_root=live_root,
        live_root=live_root,
        artifact_dir=artifact_dir,
        source_roots=source_paths,
        model=config.model,
    )
    provider_failure: str | None = None
    provider_payload_hash: str | None = None
    try:
        report_body, proposals, notes = provider.generate(source_snapshots, context)
        # Provider notes are operator signal (fail-partial drops, fallbacks). Never
        # discard them: append to the report AND echo to stderr so the cron log
        # explains a low/zero proposal count instead of leaving it a mystery.
        if notes:
            report_body = report_body.rstrip() + "\n\n## Provider notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
            for n in notes:
                print(f"[dreaming-provider-note] {n}", file=sys.stderr)
    except Exception as exc:
        provider_failure = str(exc)
        provider_payload_hash = getattr(exc, "payload_hash", None)
        report_body = _build_failure_report(
            artifact_id,
            provider_name=provider.name,
            error_message=provider_failure,
            source_count=len(source_snapshots),
            payload_hash=provider_payload_hash,
        )
        proposals = []

    artifact = DreamArtifact(
        artifact_id=artifact_id,
        created_at=_now_iso(),
        provider=provider.name,
        status="invalid" if provider_failure else "staged",
        workspace_root=str(live_root),
        source_roots=[str(path) for path in source_paths],
        report=report_body,
        sources=source_snapshots,
        proposals=proposals,
    )

    if provider_failure:
        validation_errors = [provider_failure]
        artifact.validation_errors = validation_errors
    else:
        validation_errors = validate_artifact(artifact, live_root=live_root)
        artifact.validation_errors = validation_errors
        if validation_errors:
            artifact.status = "invalid"
            if artifact.report.strip():
                validation_block = "\n".join(f"- {error}" for error in validation_errors)
                artifact.report = artifact.report.rstrip() + f"\n\n## Validation\n\n{validation_block}\n"
    if not artifact.report.strip():
        artifact.report = build_report(artifact)
    else:
        artifact.report = artifact.report.rstrip() + "\n"

    write_artifact(artifact, artifact_dir)
    return DreamCreationResult(artifact=artifact, artifact_dir=artifact_dir, validation_errors=validation_errors)


def list_artifacts(artifact_root: Path) -> list[DreamArtifact]:
    artifact_root = Path(artifact_root)
    if not artifact_root.exists():
        return []
    artifacts: list[DreamArtifact] = []
    for manifest in sorted(artifact_root.glob("*/manifest.json")):
        try:
            from .artifact import load_artifact

            artifacts.append(load_artifact(manifest.parent))
        except Exception:
            continue
    return artifacts

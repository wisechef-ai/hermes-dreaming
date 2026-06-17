from __future__ import annotations

import argparse
import os
from pathlib import Path

from .analyze import DreamRunConfig, create_dream_artifact, list_artifacts
from .apply import DreamApplyError, apply_artifact, discard_artifact
from .artifact import load_artifact
from .validation import validate_artifact


# Defaults are environment-aware so running `hermes dreaming status` from
# any cwd uses the same paths as the cron driver. Set HERMES_DREAMING_HOME
# to override the root; individual roots can be overridden too.
def _dreaming_home() -> Path:
    env = os.environ.get("HERMES_DREAMING_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes" / "dreaming"


def _default_live_root() -> Path:
    env = os.environ.get("HERMES_DREAMING_LIVE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes" / "memories"


def _default(name: str, fallback: Path) -> Path:
    env = os.environ.get(name)
    if env:
        return Path(env).expanduser()
    return fallback


def _resolve_paths() -> dict[str, Path]:
    home = _dreaming_home()
    return {
        "live_root":     _default_live_root(),
        "artifact_root": _default("HERMES_DREAMING_ARTIFACT_ROOT", home / "artifacts"),
        "source":        _default("HERMES_DREAMING_SOURCE",        home / "sources"),
        "backup_root":   _default("HERMES_DREAMING_BACKUP_ROOT",   home / "backups"),
        "archive_root":  _default("HERMES_DREAMING_ARCHIVE_ROOT",  home / "archive"),
    }


def _resolve_artifact_dir(raw: Path) -> Path:
    """Resolve an artifact argument that may be a full path OR a bare artifact id.

    The cron driver and morning brief refer to artifacts by their bare id
    (e.g. ``20260617T014529Z-140ac825``). Passing that to ``diff``/``validate``/
    ``apply``/``discard`` used to FileNotFound because the id resolved relative
    to cwd. We now fall back to ``<artifact_root>/<id>`` when the literal path
    has no manifest of its own.
    """
    raw = Path(raw)
    manifest = "manifest.json"
    # A directory that already contains the manifest is used as-is (covers
    # absolute paths and correct relative paths — preserves prior behaviour).
    if (raw / manifest).is_file():
        return raw
    # Otherwise treat the argument as a bare id under the configured root.
    candidate = _resolve_paths()["artifact_root"] / raw.name
    if (candidate / manifest).is_file():
        return candidate
    # Nothing matched — return the original so the downstream loader raises a
    # clear FileNotFoundError naming the path the user actually passed.
    return raw


def build_parser() -> argparse.ArgumentParser:
    p = _resolve_paths()
    parser = argparse.ArgumentParser(prog="dreaming", description="Hermes Dreaming MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a staged dream artifact")
    create.add_argument("--live-root", type=Path, default=p["live_root"], help="Root of the live workspace")
    create.add_argument("--artifact-root", type=Path, default=p["artifact_root"], help="Where artifacts are stored")
    create.add_argument("--source", action="append", default=None, type=Path, help="Source file or directory to scan")
    create.add_argument("--provider", default="offline-marker", help="Analysis provider to use")
    create.add_argument("--model", default=None, help="Optional provider model name")
    create.add_argument("--api-key", default=None, help="Optional provider API key")
    create.add_argument("--base-url", default=None, help="Optional provider base URL")

    diff = sub.add_parser("diff", help="Show a staged artifact")
    diff.add_argument("artifact", type=Path, help="Artifact directory")

    validate = sub.add_parser("validate", help="Validate a staged artifact")
    validate.add_argument("artifact", type=Path, help="Artifact directory")
    validate.add_argument("--live-root", type=Path, default=p["live_root"], help="Root of the live workspace")

    apply = sub.add_parser("apply", help="Apply approved changes from an artifact")
    apply.add_argument("artifact", type=Path, help="Artifact directory")
    apply.add_argument("--live-root", type=Path, default=p["live_root"], help="Root of the live workspace")
    apply.add_argument("--backup-root", type=Path, default=p["backup_root"], help="Where backups are stored")
    apply.add_argument("--approve", action="append", default=[], help="Approve a proposal id or 'all'")

    discard = sub.add_parser("discard", help="Discard a staged artifact")
    discard.add_argument("artifact", type=Path, help="Artifact directory")
    discard.add_argument("--archive-root", type=Path, default=p["archive_root"], help="Where discarded artifacts are archived")

    status = sub.add_parser("status", help="List known artifacts")
    status.add_argument("--artifact-root", type=Path, default=p["artifact_root"], help="Where artifacts are stored")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        # Default --source to the configured sources dir when omitted.
        sources = args.source
        if not sources:
            sources = [_resolve_paths()["source"]]
        result = create_dream_artifact(
            DreamRunConfig(
                live_root=args.live_root,
                artifact_root=args.artifact_root,
                source_paths=list(sources),
                provider_name=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        )
        print(f"artifact: {result.artifact_dir}")
        print(f"status: {result.artifact.status}")
        print(f"proposals: {len(result.artifact.proposals)}")
        if result.validation_errors:
            print("validation: invalid")
            for error in result.validation_errors:
                print(f"- {error}")
            return 1
        print("validation: valid")
        return 0

    if args.command == "diff":
        artifact = load_artifact(_resolve_artifact_dir(args.artifact))
        print(artifact.report.rstrip())
        if artifact.proposals:
            print()
            for proposal in artifact.proposals:
                print(f"- {proposal.id}: {proposal.target_kind} -> {proposal.target_path} [{proposal.mode}]")
                print(f"  {proposal.summary}")
        return 0

    if args.command == "validate":
        artifact = load_artifact(_resolve_artifact_dir(args.artifact))
        errors = validate_artifact(artifact, live_root=args.live_root)
        if errors:
            print("artifact is invalid")
            for error in errors:
                print(f"- {error}")
            return 1
        print("artifact is valid")
        return 0

    if args.command == "apply":
        approve_all = any(item.lower() in {"all", "*", "true", "yes"} for item in args.approve)
        approve_ids = [item for item in args.approve if item.lower() not in {"all", "*", "true", "yes"}]
        try:
            artifact = apply_artifact(
                _resolve_artifact_dir(args.artifact),
                live_root=args.live_root,
                backup_root=args.backup_root,
                approve_all=approve_all,
                approve_ids=approve_ids,
            )
        except DreamApplyError as exc:
            print(str(exc))
            return 1
        print(f"applied artifact: {artifact.artifact_id}")
        print(f"status: {artifact.status}")
        return 0

    if args.command == "discard":
        archived = discard_artifact(_resolve_artifact_dir(args.artifact), archive_root=args.archive_root)
        print(f"discarded artifact: {archived}")
        return 0

    if args.command == "status":
        artifacts = list_artifacts(args.artifact_root)
        if not artifacts:
            print("No dream artifacts found.")
            return 0
        for artifact in artifacts:
            print(f"{artifact.artifact_id}\t{artifact.status}\t{len(artifact.proposals)} proposal(s)\t{artifact.provider}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

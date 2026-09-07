# Tori fork notes — hermes-dreaming on upstream v0.4.0

**Upgraded 2026-09-07** from a pinned v0.1.0 fork (May 25, 2026) to upstream `154ce63`
(v0.4.0, Aug 2) with the Hermes-specific layer re-applied on top. Branch `main` = this.
Rollback: `git checkout fork-v010-final` (or
`~/.hermes/state/deleted-artifacts/hermes-dreaming-fork-v010-20260907.tgz`).

## What changed for the operator (essentials)

| area | before (v0.1.0 fork) | now (v0.4.0-tori.1) |
|---|---|---|
| CLI | `create diff validate apply discard status` | + `review summarize approve reject revert inbox digest report-card compact install-cron harvest providers update`; `create --from-sessions N` / `--from-since 7d`; `apply --dry-run --priority --target-kind` |
| Providers | fork-only `anthropic` (OAuth), openai-compatible | upstream `offline-marker`, `openai-compatible`, `ollama` **plus** fork `anthropic` / `anthropic-oauth` / `claude`. `hermes dreaming -- providers list` shows all four |
| Review loop | diff → apply/discard (all-or-nothing) | per-proposal `approve` / `reject --reason`; `revert` restores from backups with drift detection; `inbox --apply-ready` |
| Proposal fields | id/kind/path/mode/summary/provenance | + `risk`, `priority`, `reason`, `source_quote`, `policy_flags`; provenance must be `<file>:<line>` against numbered source lines |
| Policy (per entry) | per-entry char cap only | memory ≤240c/4 lines, user ≤220c/4, fact ≤320c, skill ≤900c/24; anti-slop; path allowlist |
| Failure mode | ANY bad proposal → whole artifact `invalid`, 0 staged | **fail-partial**: bad proposals dropped with a reason, rest staged; reasons in `REPORT.md ## Provider notes` and the cron log as `[dreaming-provider-note]` |
| facts.jsonl | N distinct rows = "conflicting proposals" (voided the night) | distinct rows additive; identical rows collapse; multi-row payloads and Opus' literal `\n` artifact accepted |
| Hermes memory write | fork `apply.py` (§-entries, budget gate, Cognee divert) | same logic, lifted verbatim into `src/hermes_dreaming/hermes_memory.py`; upstream `apply.py` delegates for `MEMORY.md`/`USER.md`/`facts.jsonl` only |
| Cron driver | `scripts/dreaming_cron.sh` | **unchanged** — `create --provider anthropic --model opus` still the call; verified live 12:57 (staged 1, posted review) |

## Hermes-specific divergences from upstream (keep on every rebase)

1. `hermes_memory.py` — §-delimited entries, whole-file injection budget (2200/1375), Jaccard ≥0.60
   near-dup drop, fact-kind → Cognee (spool when down), divert-aware verify. `atomic_write_text`
   stays owned by `apply.py` (tests monkeypatch it there).
2. `policy.target_path_is_allowed` case-folds `memory.md`/`user.md` — Hermes files are UPPERCASE.
3. `policy._entry_too_sloppy` skips "bullet entry required" for `MEMORY.md`/`USER.md` (dense prose).
4. `policy._extract_jsonl_rows` tolerant walker (real newline / whitespace / literal `\n`) used by
   policy, validation, apply. Upstream assumes one object per payload.
5. `providers._dedupe_proposals` + `validation.validate_proposals`: jsonl rows keyed by payload,
   not path. Upstream test `still_errors_on_jsonl_append_conflict` replaced by the additive pin.
6. `providers.OpenAICompatibleProvider._drop_policy_invalid` (fail-partial) — used by both the
   openai path and `anthropic_oauth_provider`.
7. `analyze.py` keeps provider notes (report + stderr). Upstream discards them.
8. `anthropic_oauth_provider.py` — Claude Max OAuth via hermes-agent adapter; opus-5 aliases;
   `temperature` 400 self-heal; registered in `build_provider` + `list_providers`.
9. `cli.py` — `HERMES_DREAMING_*` env defaults and bare-artifact-id resolution.

## Auto-update (keeps tracking upstream without losing this layer)

- Remotes: `origin` = wisechef-ai/hermes-dreaming (our fork, what `main` pushes to),
  `upstream` = asimons81/hermes-dreaming.
- Cron `dreaming-upstream-sync` (`e5dd97aec626`, Sun 04:30, no_agent, → #tori) runs
  `~/.hermes/scripts/dreaming_upstream_sync.sh`: fetch upstream → rebase `main` onto
  `upstream/main` in a throwaway worktree → run this repo's pytest → probe all 9
  divergences above → only then fast-forward `main`, tag `pre-sync-<date>`, push origin.
  Any conflict / red test / missing divergence → `[ESCALATE]` line to #tori, `main`
  untouched. Never auto-resolves conflicts. Silent when upstream has nothing new.
- Goal predicate `goals/dreaming-fork-current.md` (daily sweep) fails loudly if the sync
  stops firing (>9d), a divergence disappears, the tree goes dirty, or the nightly
  produces no valid artifact for 3 days.
- `hermes update` does NOT touch `~/.hermes/plugins/*` — this cron is the only updater.
  The 03:45 dreaming cron is a fresh process, so it picks up a synced `main` with no restart.
- Manual: `bash ~/.hermes/scripts/dreaming_upstream_sync.sh` (or `--probe` / `--self-test`).
  Rollback a bad sync: `git reset --hard pre-sync-<date>`.

## Known remaining

- (done 0907) import moved to `agent.anthropic_credentials.resolve_anthropic_token` ahead of the
  2026-09-14 compat-shim removal; prompt now states the v0.4.0 hard limits (240/220/320c, 4 lines,
  one fact per row). Live: 8 → 4 staged, 1 drop (5-line entry) — acceptable, logged.
- `hermes dreaming -- install-cron` (upstream) is NOT used; our driver is `scripts/dreaming_cron.sh`.

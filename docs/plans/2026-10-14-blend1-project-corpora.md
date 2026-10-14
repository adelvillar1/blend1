# Plan — blend1 per-project corpora + warmup freshness

```yaml
---
status: completed
created: 2026-10-14
updated: 2026-10-14
slug: blend1-project-corpora
---
```

## Context

blend1's semantic corpus is a single global store at
`~/.local/share/blend1/vectors/` (sem1 VectorStore: `<root>/<slug>/manifest.json`
+ packed float32, atomic rebuild). Two extensions, discussed and approved this
session:

1. **Per-project corpora** — facts stored per project
   (`vectors/<project>/`) so any process (not just blend1) can semantically
   search a project's facts. sem1's store is already path-parameterized plain
   files; the embedding server (llama-server :8901) is already shared local
   HTTP, so non-blend1 readers need no new service — they read the store and
   embed the query themselves.
2. **Warmup freshness** — the session-warmup ritual (branch, recaps, plans,
   git state) gains a corpus-freshness step: when work begins on a project,
   its corpus is re-derived from the project's own docs and re-embedded if
   anything changed. The manifest already carries per-entry sha256 of each
   text, so the staleness check is file reads + hashes — milliseconds, no
   embedding — and only a changed source triggers a rebuild.

The user's documentation-heavy workflow is the substrate: docs are the
canonical record; the corpus becomes the semantic index over them
(user-stated principle — docs stay canonical, this indexes, never replaces).

### Verified current-state facts (read this session)

- `corpus.py`: `STORE_ROOT` = `BLEND1_VECTORS_DIR` or
  `~/.local/share/blend1/vectors`; `seed(items)` → `VectorStore(STORE_ROOT).rebuild(slug, dims, provider, entries, meta)`;
  `retrieve(vec, k, min_score)` → `VectorStore(STORE_ROOT).load(slug)`, cosine
  top-k. Entries carry `(key, sha256(text)[:16], vec)`; meta carries the text.
- `lanes.py` `semantic_lane(prompt, retrieve=True)` calls `corpus.retrieve(res.vectors[0])`
  — no project awareness.
- `orchestrator.ask(...)` — no project param.
- `cli.py` — argparse subcommands `ask / doctor / seed`; `--live/--no-live`
  tri-state pattern exists to copy.
- `service/main.py` — `AskPayload` pydantic model; routes `/v1/health`,
  `/v1/ask`. (Pre-existing gap, NOT in scope: `BLEND1_BEARER_TOKEN` is read
  but never enforced — flag to the user, separate fix.)
- `scripts/smoke_live.py` self-seeds its corpus each run (`seed_corpus(...)`,
  `log=False`) — the persistent store is not smoke-critical.

### Documented defaults (resolved at draft; revisit at plan approval)

1. **Layout**: named project → `STORE_ROOT/<project>/<slug>/…`; project
   `None`/"default" → `STORE_ROOT` itself (the legacy store keeps working,
   zero migration for existing behavior).
2. **Resolution order** (one function `resolve_project(explicit)`):
   explicit param > `BLEND1_PROJECT` env > cwd git-repo basename > `None`
   (legacy). cwd-derivation is what makes "fresh when we begin working"
   ceremony-free; a non-repo cwd yields `None` (legacy), a repo without a
   corpus yields empty neighbors (fail-soft, same as unseeded today).
3. **Refresh source** (default, relative to cwd): `docs/**/*.md` plus root
   `CLAUDE.md`, `TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`
   when present. `--from` takes explicit files/globs and replaces the
   default. **`CLAUDE.local.md` and any non-.md file are excluded by
   construction** — the local env file carries key values and must never
   enter a store other processes read.
4. **Chunking/keying (v1)**: split each file on `^## ` headings; key =
   `<relpath>#<heading>`, text = heading + section body capped at 2,000
   chars; a non-empty preamble before the first `##` is one entry keyed
   `<relpath>#(preamble)`. Inspectable and tunable — keys are visible in
   search output.
5. **Staleness**: hash each derived text (sha256[:16], same as `seed`),
   compare the `(key, hash)` set against the store manifest; differ →
   full rebuild via the existing seed path (sem1 rebuild is whole-store);
   same → no write. Report
   `{project, source_files, items, changed, action: "rebuilt"|"unchanged"|"no-source", store}`.
6. **Legacy migration**: explicit one-time
   `blend1 corpus migrate-legacy <project>` — moves
   `vectors/<slug>/` → `vectors/<project>/<slug>/` when the legacy path
   exists and the target doesn't; idempotent no-op otherwise. Optional; the
   smoke script self-seeds, so nothing depends on it.
7. **Warmup step**: patch `~/.hermes/skills/session-warmup/SKILL.md` (the
   Hermes-native warmup skill — no approval gate; `project-warmup` left
   alone) with a fail-soft step: if `blend1` is on PATH, run
   `blend1 refresh --project <repo>`; skip silently otherwise; one summary
   line when it rebuilt.

## Approach

### Phase 1 — namespacing + resolution

1. `corpus.py`: `store_root(project)` — `None`/`"default"` → `STORE_ROOT`,
   else `STORE_ROOT/<project>`; `resolve_project(explicit=None)` per default
   #2 (cwd basename via `git rev-parse --show-toplevel`, falling back to
   `Path.cwd().name` when not a repo — cheap, no subprocess when
   `BLEND1_PROJECT` or the explicit param already decides).
2. `corpus.py`: `seed(items, *, project=None)`, `retrieve(vec, *, k=3,
   min_score=0.0, project=None)` take the resolved project.
3. `lanes.py` `semantic_lane(prompt, *, retrieve=True, project=None)` passes
   the project through. `orchestrator.ask(..., project: str | None = None)`
   resolves once and threads it to the semantic lane.
4. `cli.py`: `--project` on `ask` and `seed`; `blend1 corpus migrate-legacy
   <project>` subcommand (default #6).

### Phase 2 — refresh (docs-derived, hash-checked)

5. `corpus.py`: `derive_items(sources)` — the default source set (default #3),
   `^## ` chunking (default #4), 2,000-char cap; `refresh(*, project=None,
   sources=None)` — derive, hash-compare against the manifest, rebuild on
   diff (default #5). Never raises on a missing/empty source — `no-source`.

### Phase 3 — surfaces

6. `corpus.py`: `search(query, *, project=None, k=3)` — embed + retrieve,
   returns `[{key, score, text}]`; `__init__.py` re-exports `search` and
   `refresh_corpus` alongside `seed_corpus`.
7. `cli.py`: `blend1 search "<query>" [--project X] [--k N] [--json]` and
   `blend1 refresh [--project X] [--from PATH ...]`.
8. `service/main.py`: `POST /v1/search` — `SearchPayload {project?: str,
   query: str, k: int = 3}` → `{project, query, results, latency_ms}`.

### Phase 4 — warmup skill + tests + docs

9. `~/.hermes/skills/session-warmup/SKILL.md`: the fail-soft refresh step
   (default #7) — quoted verbatim in the implementer brief; a checklist row
   in the Warmup Checklist section.
10. Tests (offline): resolution order (param > env > cwd > legacy); two
    seeded project stores retrieve independently; `derive_items` against a
    fixture docs tree (entry count + key shape + cap); staleness
    (unchanged → no rebuild; changed → rebuild); `no-source`; `search`
    returns the seeded entry.
11. Docs: TECHNICAL-DOCUMENTATION (corpus layout + project resolution +
    refresh + `/v1/search`), FUNCTIONAL-SPECIFICATIONS (`project` param on
    ask, search CLI/service, refresh command), CLAUDE.local.md (note
    `BLEND1_PROJECT` — name only).

## Acceptance Criteria

- [ ] **AC1** — `blend1 seed --project alpha corpus.jsonl` writes
  `~/.local/share/blend1/vectors/alpha/embeddinggemma-2/manifest.json` and
  leaves the legacy `vectors/embeddinggemma-2/` store untouched (ls both
  paths).
- [ ] **AC2** — offline: `resolve_project` honors param > `BLEND1_PROJECT` >
  cwd repo basename > None (pytest with monkeypatched env + chdir).
- [ ] **AC3** — offline: with two seeded project stores, `ask(..., project=
  "alpha")` retrieves alpha's neighbor key, not beta's (pytest via
  `_lane_overrides` + real corpus stores in tmp dirs).
- [ ] **AC4** — offline: `derive_items` on a fixture docs tree yields one
  entry per `##` section (+ non-empty preambles), keys `relpath#heading`,
  every text ≤ 2,000 chars (pytest).
- [ ] **AC5** — offline: second `refresh` with unchanged sources → action
  `"unchanged"` and the manifest file's mtime unchanged; after editing a
  source file → `"rebuilt"` (pytest, tmp dirs).
- [ ] **AC6** — offline: `refresh` in a dir with no source files → action
  `"no-source"`, no store created, no raise (pytest).
- [ ] **AC7** — live: `blend1 search "churn playbook" --project blend1`
  prints the seeded entry with score + text; `--json` prints the list.
- [ ] **AC8** — live: `POST /v1/search {"project":"blend1","query":"churn
  playbook"}` on the service returns the entry with key + score.
- [ ] **AC9** — `grep -c "refresh" ~/.hermes/skills/session-warmup/SKILL.md`
  ≥ 1 and the step is fail-soft (names the `command -v blend1` guard).
- [ ] **AC10** — `uv run --extra dev pytest` green (23 existing + new) and
  `uv run --extra dev ruff check .` clean; TECHNICAL-DOCUMENTATION.md,
  FUNCTIONAL-SPECIFICATIONS.md, CLAUDE.local.md updated (grep for
  `project`/`refresh`/`/v1/search`).

## Files to be Touched

**New**
- `docs/plans/2026-10-14-blend1-project-corpora.md` (this plan)

**Modified**
- `src/blend1/corpus.py` — namespacing, resolution, derive/refresh, search
- `src/blend1/lanes.py` — `semantic_lane` project param
- `src/blend1/orchestrator.py` — `ask(project=…)`
- `src/blend1/__init__.py` — re-export `search`, `refresh_corpus`
- `src/blend1/cli.py` — `--project` flags, `search` / `refresh` /
  `corpus migrate-legacy` subcommands
- `service/main.py` — `/v1/search`
- `tests/test_corpus.py` (new file — the corpus tests, keeping
  `test_orchestrator.py` for lane logic)
- `TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`,
  `CLAUDE.local.md`
- `~/.hermes/skills/session-warmup/SKILL.md` — the fail-soft refresh step
  (outside the repo; name it explicitly in the commit)

**Deleted**
- none

## Out of Scope

- Web-lane auto-seed into the project corpus (still deferred from the
  web-lane plan).
- Cross-project search (query all projects at once).
- Non-Python client libraries (fleet etc. read the store files directly —
  documented in TECHNICAL-DOCUMENTATION, not built).
- Enforcing `BLEND1_BEARER_TOKEN` on the service (pre-existing gap; separate
  fix, flagged to the user).
- Patching the older `project-warmup` skill.

## Verification

```bash
uv sync --extra dev
uv run --extra dev pytest          # 23 existing + new corpus tests
uv run --extra dev ruff check .
# namespacing + legacy untouched
uv run blend1 seed --project alpha /tmp/alpha-corpus.jsonl
ls ~/.local/share/blend1/vectors/ ~/.local/share/blend1/vectors/alpha/
# refresh on a real project (blend1 itself, docs-derived)
cd ~/Projects/blend1 && uv run blend1 refresh --project blend1
uv run blend1 refresh --project blend1        # → action "unchanged"
uv run blend1 search "churn playbook" --project blend1
# service
service/.venv/bin/uvicorn service.main:app --port 8420 &
curl -s -X POST localhost:8420/v1/search -H 'Content-Type: application/json' \
  -d '{"project":"blend1","query":"validation gate"}'
```

Expected: refresh reports `{source_files, items, action:"rebuilt"}` the first
time and `"unchanged"` after; search returns `relpath#heading` keys with
scores; the legacy store is byte-identical after all of the above.

## Linked Artifacts

- `docs/plans/2026-10-14-blend1-web-lane.md` (completed — the auto-seed
  deferral referenced above)
- `docs/recaps/SESSION-RECAP-2026-10-14.md` (this session)
- sem1 store source: `~/Projects/sem1/src/sem1/store.py` (manifest shape,
  atomic rebuild)
- `session-warmup` skill (the step's home)

## Risks

| Risk | Mitigation |
|---|---|
| First refresh re-embeds a whole docs tree (local embedding, ~60–110 ms/item) | One-time per change; the hash check makes steady-state warmup cost file reads only. Refresh report carries item counts so the cost is visible |
| Chunking convention (h2, 2,000-char cap) is v1 — wrong granularity hurts retrieval | Keys + scores are inspectable in search output; convention documented as tunable; no silent failure mode (empty/wrong neighbors are visible in BlendResult) |
| cwd-derivation surprise: a repo named like a project gets an empty corpus | Fail-soft — empty neighbors, identical to unseeded today; `--project` and `BLEND1_PROJECT` override |
| Docs tree contains secrets (CLAUDE.local.md values) | Source set is .md-only from a named list; CLAUDE.local.md excluded by construction; corpus text never enters telemetry (house rule) |
| Warmup step adds latency to every session start | Hash check is local-file only (no embedding) when unchanged; fail-soft skip when blend1 is absent |
| Two writers to one project store | rebuild is atomic (tmp + rename); readers always see a consistent snapshot; per-project separation prevents cross-project clobber |

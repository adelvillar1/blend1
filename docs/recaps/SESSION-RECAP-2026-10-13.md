# Session Recap — 2026-10-13

## Summary

Explored and built **blend1** (`~/Projects/blend1`), a fifth lane of the sys1 house
pattern that blends the four existing providers into one `ask()`: Liquid d1 (sys1)
decides whether an ask needs generative synthesis, local EmbeddingGemma-2 (sem1) and
Prior Labs TabPFN (sdm1, cloud) precompute facts in parallel, and LFM2.5-24B-A2B
(Liquid's own API, same `LIQUID_API_KEY` as d1) synthesizes only when the router
says so. Gemini's four-tier draft was implemented on the house's real, verified
contracts — every wire in the draft was wrong in some way. All four model lanes are
live-verified end-to-end; the full blend returned a real TabPFN churn prediction
(p=0.924) synthesized by the 24B with retrieved corpus context for ≈$0.0016/ask.

The session then extended into the **specialized-model palette** program: ScaleDown
(scaledown.ai) was studied end-to-end (four models, 16 cookbook recipes, contracts
distilled into the `scaledown-models` skill), a three-way decision-model probe
(d1 vs GLiDE vs ScaleDown classify-1) assigned seats — route on any, validate on
GLiDE — and classify-1 was registered as sys1's ninth core provider. A Compress
probe on blend1's own corpus returned a negative result with a mechanism: accuracy
holds (5/5) but a billed reasoning model burns 5–20× more completion tokens on
compressed prompts, so Compress does not pay in front of the 24B leg. The user
endorsed blend1 as a keeper and set the operating principle: full-blend duration
(~15s) is not a concern — accuracy and savings justify it.

## Plans worked on

- **`docs/plans/2026-10-13-blend1-foundation.md`** — status: **completed**. All
  acceptance criteria met (parallel lanes + router gate; tabular facts as givens;
  fail-open per lane with `escalated`; corpus seed/retrieve; redacted JSONL; CLI +
  service + tests + smoke; live route verification).
- **`docs/plans/2026-10-13-blend1-validation-gate.md`** — status: **completed**
  (v0.2). GLiDE validator + two-critic escalation tier + one bounded retry.

## Commits

| Repo | Hash | Message |
|------|------|---------|
| blend1 | `34b4126` | blend1 v0.1.0: the blended lane — d1 router + EmbeddingGemma-2 + TabPFN + LFM2.5-24B |
| blend1 | `d597b6d` | blend1 v0.2: validation gate — GLiDE rates the 24B, critics break ties, one retry |
| blend1 | `00fa252` | smoke: exercise the validation gate end-to-end |
| blend1 | `9900774` | docs: record the user endorsement |
| blend1 | `3977005` | docs: Compress probe result — not a fit for the 24B leg |
| sem1 | `b7bd58b` | fix: a bare string is one text item, not an iterable of characters |
| sys1 | `701b44a` | feat: ScaleDown classify-1 provider (the Jev mirror) + MCP pass-through; stale safety-routing test fixed |

## What was added

- `src/blend1/` — orchestrator (parallel lanes + route gate + merge), five lane
  callables, corpus store, BlendResult contract, env loader, redacted telemetry, CLI.
- `service/` — FastAPI wrapper (`/v1/ask`, `/v1/health`, port 8420).
- `tests/test_orchestrator.py` — 15 offline tests (stub lanes).
- `scripts/smoke_live.py` — live smoke (doctor + seed + three asks).
- **v0.2 validation gate** — `validate_lane` (GLiDE task `blend_validate`,
  d1 on oversized states), `escalate_lane` (qwen3.8-flash → glm-5.3-flash critic
  chain from the Hermes agent config), deterministic `number_check`, one
  bounded retry with verdict-as-feedback, `validation` field on BlendResult.
- Methodology artifacts: `CLAUDE.md`, `CLAUDE.local.md` (gitignored),
  `TECHNICAL-DOCUMENTATION.md`, `FUNCTIONAL-SPECIFICATIONS.md`, this recap,
  plan docs with frontmatter, full `.gitignore`.
- **Palette program (later in the session):** the `scaledown-models` skill
  (four model contracts, 16 cookbook recipes, problem shapes, wire gotchas);
  `SCALEDOWN_API_KEY` stored in `~/.config/blend1/env` (chmod 600); the
  three-way probe artifacts (`/tmp/three_way_probe.py`); the Compress probe
  (`/tmp/compress_probe.py`); sys1's `scaledown` provider (recap in sys1).

## What was fixed

- **sem1 (upstream, committed):** `normalize_items` iterated a bare string into
  single-character items — `embed("hello")` sent 5 items (48× server work on real
  prompts, wrong counts) despite the docstring promising str acceptance. Root cause:
  no str special-case before the `for it in items` loop. Fixed + regression test;
  sem1 suite 20/20 green.
- **blend1 corpus:** `seed()` stored under the server's full model path while
  `retrieve()` looked up the short slug — the store was never found (neighbors
  always empty). Fixed: store under the fixed slug; the real model id rides in the
  manifest.
- **blend1 generate lane:** rewired from OpenRouter to Liquid's own API
  (`/openai/v1/chat/completions`, 404 on `/v1/chat/completions`), one key for both
  Liquid lanes; reasoning-model handling (empty content = failed leg); cost +
  tok/s telemetry; output cap raised to 16K per user (128K context window).
- **blend1 number check (two rounds):** percent-vs-probability formatting
  (`92.4%` vs `p=0.924` — same number, now canonicalized); and the givens set
  excluded the prompt, so the asker's own numbers (customer ids) false-positived
  — the critic chain overruled it live, then the allowed set gained the prompt.
- **sys1 (committed):** the stale safety-routing test (see sys1's recap).

## Files changed

- New repo: blend1 (16 files, 1581 insertions) + methodology artifacts.
- sem1: `src/sem1/types.py`, `tests/test_sem1.py`.
- Config: `~/.config/sdm1/env` created (chmod 600) from the key documented in
  `~/Projects/sdm1/CLAUDE.local.md` — its canonical home remains that file.

## Doc updates applied

- README lane table + "what the draft got wrong" table corrected to the verified
  Liquid-API reality; plan live-state table updated (all four model lanes live).

## Doc updates deferred

- None. CLAUDE.md "Today's state" carries the live-verified state.

## Open questions / next steps

1. **Gate floor fitting** — the 0.70 noul floors are UNFITTED house defaults.
   The machinery ships (verdicts + raw nouls in the JSONL row); fitting waits
   for a feedback stream. **User endorsement 2026-10-13: blend1 is a keeper —
   the ~15s full-blend duration is not a concern; accuracy and savings
   justify it.** That settles the latency-vs-completeness tradeoff in favor
   of the complete gated blend — do not "optimize" the gate or the tabular
   fit away.
2. **`decompose` path** — `answer_shape="decompose"` is recorded but not acted
   on (recursive split is a v0.3 decision).
3. **Local 24B fallback** — LFM2-24B-A2B is open weights; a quantized GGUF on
   the existing llama.cpp build needs more than this machine's 16 GB unified
   memory.
4. **MRL truncation** — the draft's 128d truncation is unverified on this
   build; the lane keeps full 768d until probed.
5. **Neighbor relevance gating** — the gate now catches the drift irrelevant
   neighbors cause (the cheaper fix landed); a relevance floor on injection
   remains the complementary option.
6. **Live retry cycle** — proven offline; the organic drift observed earlier
   today rates 0.30 on GLiDE (clear fail → retry path), but the retry has not
   fired in a live run yet.
7. **ScaleDown seats** — classify-1 is admitted as a route seat on one probe;
   its floors stay unfitted until the JSONL stream says otherwise. Compress
   probed and rejected for the 24B leg (see the Compress probe in the skill).
8. **Git** — all repos committed; blend1 published to GitHub, sys1 pushed on
   `feature/plan-surface`, sem1 pushed on `main` (carrying three pre-existing
   unpushed MCP commits alongside the sem1 fix).

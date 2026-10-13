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

---

## Part 2 — the specialized-model palette program (appended at wrap-up)

The session continued past the v0.2 gate into the user's strategic frame
(recorded in memory): *combining specialized models to raise the quality of the
dev process* — with the user's stated interest being **enhancing reasoning at a
lower cost**.

### Commits (part 2)

| Repo | Hash | Message |
|------|------|---------|
| sys1 | `701b44a` | feat: ScaleDown classify-1 provider (the Jev mirror) + MCP pass-through; stale safety-routing test fixed |
| sys1 | `b033e4f` | docs: session recap 2026-10-13 |
| blend1 | `3977005` | docs: Compress probe result — not a fit for the 24B leg |
| blend1 | `73822a6` | docs: reasoning-at-lower-cost probe |
| blend1 | `5d4ec60` | docs: reframe the probe economics — capacity preservation, not cost-per-call (user correction) |
| blend1 | `6e70459` | docs: Julia-seat probe |
| blend1 | `0f15fb7` | docs: correct the gate-cost claim to the user's Fastino receipt |
| — | — | blend1 repo created public (github.com/adelvillar1/blend1) and pushed |

### What was added (part 2)

- sys1's `scaledown` provider (classify-1, the documented Jev mirror) — full
  detail in sys1's own recap; admitted as a **route seat** on probe evidence.
- The `scaledown-models` skill (four model contracts, 16 cookbook recipes,
  problem shapes, wire gotchas).
- Three probes on blend1's own 21K-token corpus with deterministic grading
  against source-verified ground truth:
  1. **Compress** — accuracy held 5/5 but the economics invert on a billed
     reasoning model (5–20× completion burn; a 3% kept ratio dropped exact
     figures an arithmetic question needed). **Rejected** — three probes, three
     stacks, same direction.
  2. **Reasoning-at-lower-cost** — qwen3.8-flash (token-plan) **matched the
     24B 5/5** on reasoning-shaped questions (arithmetic over documented
     costs, multi-hop over the orchestrator's control flow, gate semantics).
  3. **Julia-seat** — qwen holds a *decision* seat: route 4/4 with margins
     sharper than every paid seat (0.0 vs d1's 0.002; 1.0 vs glide's 0.994),
     gate outcomes 4/4 (DRIFT's contra=1.0 is a clear fail → retry), 0 parse
     failures. Caveats: 7–43s latency; 4 designed states.

### User corrections absorbed (durable)

- **The economics frame is CAPACITY, not cost-per-call** (stated 2026-10-13):
  the win of a cheap model matching a frontier one is preserved frontier
  budget for the harder problems — never re-frame as a per-call comparison.
- **Blended estimates suffice on cost questions**; do not chase per-call
  precision and do not instrument per-request cost measurement.

### Vendor receipts (the user's own meters)

- Fastino/GLiDE: $2.82 / 5.7K requests / 9.2M tokens over 30 days —
  "almost too cheap to measure."
- Julia-1: $0.000239525 / 13 calls.
- Conclusion: the decision layer is pocket change at personal scale; **all
  real cost pressure lives in the synthesis seat** — the seat the almost-free
  model holds at parity. The old docs' gate-cost range was never instrumented
  and was corrected to the blended statement.

### PARKED — the next rung (not started)

**Cheap synthesizer by default, the validation gate as allocator, 24B reserved
for gate-fails and decompose-shaped asks.** All three probe legs (parity,
decision-seat, economics) are behind it; the build is a `synthesizer` seat in
the generate lane with fail-open, per the lane pattern. **The user has other
plans gated on the Fleet session finishing first — do not start this rung
until Fleet's work is confirmed complete.** The verify-stage experiment (dev-
process application: acceptance criteria as noul heads, providers rating the
same artifact) remains the follow-on thread.

### Wrap-up debts (2026-10-13)

- **sys1 CLAUDE.md "Today's state" is stale**: the roster line still says 8
  core providers (scaledown makes 9) and the test count says 364 (now 369).
  The write was blocked by the CLAUDE.md protection gate (approval prompt
  timed out — user AFK). Retry the edit when the user is present.

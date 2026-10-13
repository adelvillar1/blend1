---
status: completed
created: 2026-10-13
updated: 2026-10-13
slug: blend1-validation-gate
---

# Plan: blend1 validation gate (v0.2)

## Context

v0.1 routes (d1) → precomputes (sem1 + sdm1) → synthesizes (lfm2.5-24b) with
facts injected as givens. Nothing rated the 24B's answer. Observed live
2026-10-13: when retrieved neighbors were irrelevant, the 24B answered from
the context instead of the question (recap open question #6). A tail-end
validation gate catches exactly that class of failure, and the retry cycle
turns a caught failure into a corrected answer.

Live probe (2026-10-13, sys1 venv, GLiDE + d1 on a good answer vs. the
observed drift) decided the design:

- **GLiDE (glide, Fastino) is the validator** — margins decisive on the
  discriminating head (GOOD 0.80 / DRIFT 0.30) where d1's DRIFT 0.62 sits
  dangerously near the 0.70 floor; also a different vendor from the d1
  router, so the critic is not the model that framed the task.
- **d1 is the fallback** for states that exceed GLiDE's 40K rendered-prompt
  limit (a 16K-token answer + facts + question can); d1's window is 65,536.
- The probe **broke the naive design**: a `consistent_with_facts` noul worded
  as "invents nothing beyond the facts" fails GOOD answers (0.32) because
  good answers legitimately expand beyond `(none given)`. The gating head
  must check **contradiction**, not expansion.
- GLiDE's score-head confidence is ~0.008 (documented glide quirk) — the
  score head records, it does not gate.

## Approach

New `validate` lane + escalation tier + one bounded retry in the orchestrator:

1. **Task `blend_validate`** (sys1 Task, run through glide → d1 on size):
   - `answers_the_question` (noul, floor 0.70) — **gates**
   - `contradicts_facts` (noul, floor 0.70, polarity P(contradicts)) —
     **gates**, worded as contradiction, not expansion
   - `answer_quality` (score 1–5 with legend) — telemetry only (glide's
     score confidence is too low to gate; `expected_level` is recorded)
2. **Deterministic number check in code** — every metric-shaped number in
   the answer (percentages canonicalized to probabilities: 92.4% == p=0.924)
   must appear in the givens = prompt + TabPFN facts + semantic facts. The
   prompt is in the allowed set because the asker's own numbers (customer
   ids, dates) are legitimately quotable — the check catches numbers the
   MODEL invents. List markers and small ordinals are excluded by shape.
3. **Three outcomes** (user directive 2026-10-13: inconclusive escalates):
   - clear pass (both nouls clear, numbers ok) → ship
   - clear fail (a noul a full 0.15 band beyond its floor) → one retry
   - **inconclusive** (near-floor nouls, mixed signals, or a flagged number
     under passing nouls) → **escalate to the critic chain**:
     qwen3.8-flash (Token Plan, `TOKEN_PLAN_API_KEY`) → glm-5.3-flash
     (ollama-cloud, `OLLAMA_API_KEY`), both wired from the Hermes agent
     config and verified live. No critic reachable → ship ungated with the
     uncertainty recorded — the gate NEVER blocks.
4. **One retry cycle** — on definitive failure, re-run the generate lane
   with the verdict injected as feedback. Max one retry (worst case ≈ 2×
   generate cost, failures only). A failed retry ships the ORIGINAL answer
   flagged (`retry_passed: false`, `escalated: true`).
5. **Calibration from day one** — every outcome + both noul values + the
   number-check result ride in the JSONL row; floors are UNFITTED house
   defaults until feedback rows exist.

## Acceptance Criteria

- [x] `validate_lane(state, givens=)` runs the `blend_validate` task through
      glide, falling back to d1 on oversized states; returns a report, never
      raises.
- [x] Number check is deterministic with percent/probability canonicalization
      and list-marker exclusion; flagged numbers name themselves.
- [x] Three outcomes: passed / failed / inconclusive→escalated; escalation
      runs the two-critic chain and records which critic decided.
- [x] Definitive failure triggers exactly one retry with the verdict
      injected; retry failure ships the original flagged, never blocks.
- [x] Offline suite: 15 tests — pass→no retry, fail→one retry with feedback
      naming the head, retry success ships corrected, inconclusive→no retry
      + escalated, gate down→ungated, number-check units, critic-chain
      missing-key reporting.
- [x] Live evidence: good case passes clean (atq 0.995, numbers ok); the
      mixed-signal false positive (customer id 4471 not in givens) escalated
      and the critic overruled it; probe values remain the drift acceptance
      evidence (glide 0.30 → clear fail).

## Out of scope

- Multi-retry loops / self-consistency re-asks (band first, then decide).
- Fitting the floors from feedback rows (machinery ships; fitting waits).
- Gating the direct path (composed from facts by code — nothing to judge).
- Dual decision-model critics (d1 + glide both) — d1 stays the size-gate
  fallback only.

## Verification

- `uv run --extra dev pytest` — 15 offline tests green, ruff clean.
- Live: good case clean pass; false-positive escalation overruled by the
  critic chain; non-drift declined-answer pass (atq 0.893).

## Linked artifacts

- `TECHNICAL-DOCUMENTATION.md` — validate lane wire, heads, critic chain
- `FUNCTIONAL-SPECIFICATIONS.md` — gate + escalation + retry behavior
- `docs/recaps/SESSION-RECAP-2026-10-13.md` — session journal

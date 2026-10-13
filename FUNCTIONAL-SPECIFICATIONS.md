# blend1 — Functional Specifications

> Behavior contract: what `ask()` guarantees, how routing decides, what fail-open
> means. Companion to `TECHNICAL-DOCUMENTATION.md` (wires and modules).

## The one-sentence contract

`blend1.ask(prompt, table=…, target=…)` returns one answer composed from the lanes
that could answer — with the 24B generative leg paid for **only** when Liquid d1
judges the ask needs open-ended synthesis.

## ask() behavior

### Inputs

| Arg | Type | Meaning |
|---|---|---|
| `prompt` | str | the user's text |
| `table` | list[dict] \| None | tabular rows; rows with non-null `target` are context, null-target rows are the query |
| `target` | str \| None | the column TabPFN predicts — required when `table` is given |
| `task_type` | str | `"classification"` (default) or `"regression"` |
| `voice` | bool | also render the final answer to audio via gen1 (default False — spends money) |
| `retrieve` | bool | retrieve semantic corpus neighbors (default True) |
| `log` | bool | append the redacted JSONL row (default True) |

### Routing decision (the cost gate)

1. d1 answers the `blend_route` task: noul `needs_generation` (P that the ask needs
   synthesis beyond a fact/metric/classification) + choice `answer_shape`.
2. **Gate:** `needs_generation ≥ 0.70` → synthesized path (24B called).
   Below → direct path (**no 24B call, no generative spend**).
3. `answer_shape == "decompose"` is recorded in `path` (the ask bundles sub-asks) —
   v0.2 will act on it; today it routes like synthesized.
4. **Router unavailable → fail toward synthesis** (quality over cost); the error
   rides in the result and `escalated` is True.

### Answer composition

- **Synthesized + generator ok:** the 24B's text, with the exact TabPFN facts and
  top-k corpus neighbors injected into its system prompt as givens.
- **Direct, or generator failed open:** composed from computed facts only —
  TabPFN predictions rendered exactly, nearest corpus keys with cosines, and (when
  the generator failed) the reason it failed. Facts stay exact; the prose is simply
  absent.

### The validation gate (v0.2)

Every synthesized answer is rated before it ships, by **GLiDE** (a decision
model from a different vendor than the d1 router, so the critic is not the
model that framed the task). The gate has three outcomes:

| Outcome | When | What happens |
|---|---|---|
| `passed` | both nouls clear their floors, numbers check out | the answer ships |
| `failed` | a noul sits a full 0.15 band beyond its floor (clear drift/contradiction) | **one retry** with the verdict injected as feedback |
| `inconclusive` | near-floor nouls, mixed signals, or a flagged number under passing nouls | **escalate to the critic chain** (qwen3.8-flash → glm-5.3-flash); the critic's verdict decides |

Escalation details:

- The critic sees the same state (question + facts + answer) and returns
  `{"passed": bool, "reason": str}`. Which critic decided rides in the result
  (`validation.critic`).
- The chain walks until one critic answers; missing keys and dead endpoints
  record their reason and the walk continues.
- **No critic reachable → the answer ships ungated** with
  `outcome: "inconclusive"` and `escalated: true` — the caller knows the
  answer was never cleared. The gate never blocks.

The retry cycle:

- Exactly one retry, only on a definitive `failed`. The verdict is injected
  into the generate lane's system prompt as an explicit correction ("A
  validation gate rejected your previous answer: your answer drifted from the
  question that was asked (rated 0.21 …) — answer the actual ask.").
- The retry is gated too. If it clears, the corrected answer ships
  (`retried: true`). If not, the **original** answer ships flagged
  (`retry_passed: false`, `escalated: true`) — never blocked, never silent.
- Worst case ≈ 2× generate cost, on failures only. Passing answers cost one
  d1-class rating (~$0.00001) — a rounding error on the $0.0015 leg it guards.

The deterministic number check (code, not a model):

- Every metric-shaped number in the answer — decimals, percentages, ≥2-digit
  integers; list markers and small ordinals excluded by shape — must appear
  in the givens (prompt + TabPFN facts + semantic facts).
- Percentages canonicalize to probabilities, so the fact's `p=0.924` and the
  answer's `92.4%` compare equal. A cited-but-absent number is flagged by
  name in `validation.flagged_numbers`.

### Guarantees

1. **Facts are never invented.** TabPFN's number reaches the 24B as a given; the
   system prompt forbids altering it. If the facts are insufficient, the model is
   instructed to say so (observed live: it does).
2. **Direct asks cost $0 in model spend.** No generative call is made.
3. **No lane failure breaks the ask.** Every lane fails open with its reason named;
   `BlendResult.escalated` is True when any lane errored.
4. **Nothing sensitive is logged.** JSONL rows carry sha256 + counts + telemetry —
   never prompt text, answer text, or key values.
5. **The router's decision is auditable.** The lane report carries the raw noul
   probability, the shape distribution, and d1's per-call cost.
6. **A rated answer is either cleared or flagged.** The validation gate's
   outcome (`passed` / `failed` / `inconclusive` / `escalated_*`) rides in the
   result; an answer that never cleared the gate ships with
   `escalated: true`, never silently. The gate's floors are UNFITTED house
   defaults (0.70) — treat a failure as "worth one retry", not "wrong",
   until feedback rows fit them.

### Failure semantics per lane

| Lane | Unconfigured/unreachable → |
|---|---|
| route | lane report `status="error"` with sys1's reason; blend defaults to synthesis |
| semantic | `status="error"` (endpoint down / sem1 import failed); neighbors omitted |
| tabular | sdm1's fail-open: `verdict="error"` naming `TABPFN_API_KEY`; facts omitted |
| generate | `status="error"` naming the key / HTTP status / unreachable; direct-style fallback answer |
| validate | `status="error"` (GLiDE down) → the answer ships **ungated**; a definitive fail triggers the retry; inconclusive escalates to the critic chain |
| critics | all unreachable → `outcome="inconclusive"`, answer ships ungated with `escalated: true` |
| voice | `status="error"` naming the missing extra; audio omitted |

## CLI

```bash
blend1 ask "What is the churn probability for customer 4471?" --table churn.csv --target churned
blend1 ask "Write a save email for customer 4471." --table churn.csv --target churned --json
blend1 ask "Read this back to me." --voice          # spends money, on purpose
blend1 ask "…" --no-retrieve                        # skip corpus retrieval
blend1 doctor                                       # per-lane availability, reasons named
blend1 seed corpus.jsonl                            # [{"key":…,"text":…}, …]
```

`--json` prints the full BlendResult. Without it, stdout is the final answer and a
stderr line notes escalation when a lane failed open.

## HTTP service

```bash
# from the PROJECT ROOT (namespace package):
service/.venv/bin/uvicorn service.main:app --port 8420
```

| Method | Path | Body |
|---|---|---|
| `GET` | `/v1/health` | — → per-lane availability snapshot |
| `POST` | `/v1/ask` | `{prompt, table?, target?, task_type?, voice?, retrieve?}` → BlendResult |

Set `BLEND1_BEARER_TOKEN` to require `Authorization: Bearer …` (health stays open,
like sys1's service).

## Live behavior reference (verified 2026-10-13)

| Ask | d1 noul | Path | Lanes that answered |
|---|---|---|---|
| "What is the churn probability for customer 4471?" | 0.002 | direct | route, semantic — **no 24B call** |
| "Write a one-paragraph save-strategy note…" | 0.978 | synthesized | route, semantic, generate |
| "Based on this customer data, what should we do about 4471?" + table | 0.743 | synthesized | route, semantic, **tabular (p=0.924)**, generate |

Cost per synthesized ask ≈ $0.0016 (d1 $0.000004 + TabPFN metered + 24B);
per direct ask ≈ $0.000004. The validation gate is metered on Fastino's
decision models, which return no cost field (GLiDE's `cost_usd` is always
None), so treat any per-call figure as a blended estimate: the user's 30-day
Fastino receipt (5.7K requests, 9.2M tokens, $2.82) blends to ~$0.0005/request
and ~$0.31/1M tokens — sub-cent per gate call at any personal scale, and never
a reason to skip the gate. A definitive failure adds at most one extra
generate call.

### Gate outcomes observed live (2026-10-13)

| Ask | Gate outcome | Detail |
|---|---|---|
| churn ask + table | `passed` | atq 0.995, contra 0.003, numbers ok — no retry, no escalation |
| same, first run | `escalated_passed` | number check false-positived on the customer id (fixed: givens now include the prompt); the critic chain overruled and passed |
| onboarding note + irrelevant corpus | `passed` | atq 0.893 — the 24B correctly declined to use irrelevant facts, and the gate agreed |

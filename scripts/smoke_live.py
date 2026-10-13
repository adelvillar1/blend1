"""Live smoke: exercise every lane that can answer RIGHT NOW, with real calls.

Run:  uv run --project . python scripts/smoke_live.py     (from blend1/)

Expected on a machine with only the liquid + llama-server lanes keyed:
  route    → ok   (d1 decides per prompt)
  semantic → ok   (768d embeddings from 127.0.0.1:8901)
  tabular  → error (names TABPFN_API_KEY — the honest fail-open)
  generate → error (names OPENROUTER_API_KEY — the honest fail-open)
The blend still returns a composed answer and marks itself escalated.
"""

import json

import blend1

PROMPTS = [
    ("direct", "What is the churn probability for customer 4471?", None, None),
    ("synthesized", "Write a one-paragraph save-strategy note for customer 4471 given their churn risk.", None, None),
]

TABLE = [
    {"logins_30d": 10, "tickets_90d": 0, "spend": 150, "churned": 0},
    {"logins_30d": 2, "tickets_90d": 5, "spend": 20, "churned": 1},
    {"logins_30d": 12, "tickets_90d": 1, "spend": 200, "churned": 0},
    {"logins_30d": 1, "tickets_90d": 8, "spend": 10, "churned": 1},
    {"logins_30d": 8, "tickets_90d": 2, "spend": 120, "churned": 0},
    {"logins_30d": 3, "tickets_90d": 6, "spend": 35, "churned": 1},
    {"logins_30d": 1, "tickets_90d": 6, "spend": 15, "churned": None},
]


def main() -> int:
    print("== doctor ==")
    print(json.dumps(blend1.doctor(), indent=2))

    # Seed a tiny semantic corpus so retrieval is exercised live.
    print("\n== seed corpus ==")
    print(json.dumps(blend1.seed_corpus([
        {"key": "billing-errors-faq", "text": "billing errors cluster around invoice dates and prorated upgrades"},
        {"key": "churn-playbook", "text": "customers with falling login frequency and rising ticket volume churn within 60 days"},
        {"key": "lag-tickets", "text": "server lag complaints spike after the 4.2 release"},
    ]), indent=2))

    for label, prompt, table, target in PROMPTS:
        print(f"\n== ask [{label}] {prompt!r} ==")
        r = blend1.ask(prompt, table=table, target=target, log=False)
        print(json.dumps(r.as_dict(), indent=2, default=str)[:2200])

    print("\n== ask [tabular lane, real sdm1 call] ==")
    r = blend1.ask(
        "What is the churn probability for customer 4471?",
        table=TABLE,
        target="churned",
        log=False,
    )
    print(json.dumps(r.as_dict(), indent=2, default=str)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

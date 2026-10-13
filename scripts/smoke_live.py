"""Live smoke: exercise every lane that can answer RIGHT NOW, with real calls.

Run:  uv run python scripts/smoke_live.py     (from blend1/)

The corpus is seeded with entries that are IRRELEVANT to most asks, so the
semantic lane is exercised under drift pressure (the failure mode the v0.2
validation gate exists to catch). Every lane that can answer answers for
real; the gate's outcome rides in the JSON under "validation".
"""

import json

import blend1

PROMPTS = [
    ("direct", "What is the churn probability for customer 4471?", None, None),
    ("synthesized", "Write a one-paragraph save-strategy note for customer 4471 given their churn risk.", None, None),
]

TABLE = [
    {"logins_30d": 10, "tickets_90d": 0, "spend": 150, "churned": 0},
    {"logins_30d": 2,  "tickets_90d": 5, "spend": 20,  "churned": 1},
    {"logins_30d": 12, "tickets_90d": 1, "spend": 200, "churned": 0},
    {"logins_30d": 1,  "tickets_90d": 8, "spend": 10,  "churned": 1},
    {"logins_30d": 8,  "tickets_90d": 2, "spend": 120, "churned": 0},
    {"logins_30d": 3,  "tickets_90d": 6, "spend": 35,  "churned": 1},
    {"logins_30d": 1,  "tickets_90d": 6, "spend": 15,  "churned": None},
]


def _show(r) -> None:
    d = r.as_dict()
    v = d.get("validation") or {}
    gate = (
        f"gate={v.get('outcome')} passed={v.get('passed')} "
        f"atq={round(v.get('answers_the_question') or 0, 2)} "
        f"contra={round(v.get('contradicts_facts') or 0, 2)} "
        f"numbers={v.get('number_check')} critic={v.get('critic')}"
        if v else "gate=not-run (direct path)"
    )
    print(f"path={d['path']} {gate} escalated={r.escalated} "
          f"latency={d['total_latency_ms']}ms")
    print(d["final_answer"][:600])


def main() -> int:
    print("== doctor ==")
    print(json.dumps(blend1.doctor(), indent=2))

    # Seed a tiny semantic corpus so retrieval is exercised live. These
    # entries are deliberately irrelevant to most asks — drift pressure.
    print("\n== seed corpus ==")
    print(json.dumps(blend1.seed_corpus([
        {"key": "billing-errors-faq", "text": "billing errors cluster around invoice dates and prorated upgrades"},
        {"key": "churn-playbook", "text": "customers with falling login frequency and rising ticket volume churn within 60 days"},
        {"key": "lag-tickets", "text": "server lag complaints spike after the 4.2 release"},
    ]), indent=2))

    for label, prompt, table, target in PROMPTS:
        print(f"\n== ask [{label}] {prompt!r} ==")
        _show(blend1.ask(prompt, table=table, target=target, log=False))

    print("\n== ask [tabular lane, real sdm1 call] ==")
    _show(blend1.ask(
        "What is the churn probability for customer 4471?",
        table=TABLE, target="churned", log=False,
    ))

    print("\n== ask [full blend + validation gate: facts-relevant synthesis] ==")
    _show(blend1.ask(
        "Based on this customer data, what should we do about customer 4471?",
        table=TABLE, target="churned", log=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

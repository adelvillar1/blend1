"""blend1 CLI: ask / doctor / seed.

    blend1 ask "what should we do about customer 4471?" --table churn.csv --target churned
    blend1 ask "write a save email for customer 4471" --json
    blend1 doctor
    blend1 seed corpus.jsonl        # [{"key":..., "text":...}, ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="blend1", description="The blended lane: d1 router + EmbeddingGemma-2 + TabPFN + LFM2-24B")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="one blended ask")
    p_ask.add_argument("prompt")
    p_ask.add_argument("--table", help="CSV with labeled context rows + unlabeled query rows")
    p_ask.add_argument("--target", help="column TabPFN predicts (required with --table)")
    p_ask.add_argument("--task-type", default="classification")
    p_ask.add_argument("--voice", action="store_true", help="also render the answer with gen1 speak")
    p_ask.add_argument("--no-retrieve", action="store_true", help="skip semantic corpus retrieval")
    p_ask.add_argument("--live", action="store_true", help="force the web lane on regardless of the router")
    p_ask.add_argument("--no-live", action="store_true", help="suppress the web lane regardless of the router")
    p_ask.add_argument("--json", action="store_true", help="print the full BlendResult")

    sub.add_parser("doctor", help="per-lane availability snapshot")

    p_seed = sub.add_parser("seed", help="seed the semantic corpus")
    p_seed.add_argument("file", help="JSONL of {\"key\":..., \"text\":...} items")

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        import blend1

        print(json.dumps(blend1.doctor(), indent=2))
        return 0

    if args.cmd == "seed":
        import blend1

        items = []
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        print(json.dumps(blend1.seed_corpus(items), indent=2))
        return 0

    # ask
    import blend1

    table = _read_csv(args.table) if args.table else None
    if table is not None and not args.target:
        print("error: --target is required with --table", file=sys.stderr)
        return 2
    # CSV cells arrive as strings; let sdm1's coercion handle booleans/numbers
    # by round-tripping through its own reader semantics.
    live = True if args.live else (False if args.no_live else None)
    result = blend1.ask(
        args.prompt,
        table=table,
        target=args.target,
        task_type=args.task_type,
        voice=args.voice,
        retrieve=not args.no_retrieve,
        live=live,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    else:
        print(result.final_answer)
        if result.escalated:
            print("\n[escalated: a lane failed open — see --json]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

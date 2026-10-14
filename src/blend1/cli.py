"""blend1 CLI: ask / doctor / seed / search / refresh / corpus.

    blend1 ask "what should we do about customer 4471?" --table churn.csv --target churned
    blend1 ask "write a save email for customer 4471" --json
    blend1 doctor
    blend1 seed corpus.jsonl        # [{"key":..., "text":...}, ...]
    blend1 search "churn playbook" --project blend1
    blend1 refresh --project blend1            # docs-derived, hash-checked
    blend1 corpus migrate-legacy blend1        # move the legacy store under a project

"""

from __future__ import annotations

import argparse
import csv
import json
import sys

_PROJECT_HELP = "corpus project (default: $BLEND1_PROJECT or the cwd repo name)"


def _corpus_guard(fn) -> int:
    """Run a corpus-touching command with typed failures as clean one-line
    errors — never a traceback. The warmup step's fail-soft contract depends
    on this: a down embedding server or a bad project name is a message and
    a non-zero exit, not a stack trace in the middle of a session start."""
    from sem1 import Sem1Error

    try:
        fn()
        return 0
    except ValueError as e:  # bad project name, unusable source path
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Sem1Error as e:  # embedding server unreachable/misbehaving
        print(f"error: embedding server unavailable: {e}", file=sys.stderr)
        return 1
    except OSError as e:  # unreadable source, unwritable store
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


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
    p_ask.add_argument("--project", help=_PROJECT_HELP)
    p_ask.add_argument("--json", action="store_true", help="print the full BlendResult")

    sub.add_parser("doctor", help="per-lane availability snapshot")

    p_seed = sub.add_parser("seed", help="seed the semantic corpus")
    p_seed.add_argument("file", help="JSONL of {\"key\":..., \"text\":...} items")
    p_seed.add_argument("--project", help=_PROJECT_HELP)

    p_search = sub.add_parser("search", help="semantic search over a project's corpus")
    p_search.add_argument("query")
    p_search.add_argument("--project", help=_PROJECT_HELP)
    p_search.add_argument("--k", type=int, default=3, help="neighbors to return (default 3)")
    p_search.add_argument("--json", action="store_true", help="print the full result list as JSON")

    p_refresh = sub.add_parser("refresh", help="re-derive the corpus from docs; rebuild only when changed")
    p_refresh.add_argument("--project", help=_PROJECT_HELP)
    p_refresh.add_argument("--from", dest="sources", nargs="*", default=None,
                           help="explicit source files/globs (replaces the default docs set)")

    p_corpus = sub.add_parser("corpus", help="corpus maintenance")
    corpus_sub = p_corpus.add_subparsers(dest="corpus_cmd", required=True)
    p_migrate = corpus_sub.add_parser(
        "migrate-legacy", help="move the legacy store under a project (idempotent)"
    )
    p_migrate.add_argument("project")

    args = ap.parse_args(argv)
    return _corpus_guard(lambda: _dispatch(args))


def _dispatch(args) -> int:
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
        print(json.dumps(blend1.seed_corpus(items, project=args.project), indent=2))
        return 0

    if args.cmd == "search":
        import blend1

        results = blend1.search(args.query, project=args.project, k=args.k)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            if not results:
                print("(no neighbors)", file=sys.stderr)
            for r in results:
                print(f"{r['score']:.4f}  {r['key']}")
                print("\n".join(f"    {line}" for line in r["text"].splitlines()))
        return 0

    if args.cmd == "refresh":
        import blend1

        report = blend1.refresh_corpus(project=args.project, sources=args.sources)
        print(json.dumps(report, indent=2))
        return 0

    if args.cmd == "corpus":
        from blend1.corpus import migrate_legacy

        if args.corpus_cmd == "migrate-legacy":
            print(json.dumps(migrate_legacy(args.project), indent=2))
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
        project=args.project,
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

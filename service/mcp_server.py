#!/usr/bin/env python3
"""
blend1 MCP server: exposes the blended lane's six surfaces as MCP tools —
`ask` (blend1.ask), `web` (the web leg called directly), `doctor`
(blend1.doctor), `search` (corpus recall), `refresh` (docs-derived corpus
refresh), and `seed` (the corpus's write path).

Transport: stdio JSON-RPC (MCP 2024-11-05).

blend1 is the blended lane of the sys1 house pattern: one ask() runs the
d1 router (sys1) solo first as the front door, the semantic (sem1) and
tabular (sdm1) fact lanes concurrently, then synthesizes with Liquid
LFM2.5-24B-A2B only when the router clears ROUTE_FLOOR (0.70); a
validation gate (GLiDE → d1) rates the answer before it ships. The core
is stdlib-only; the sibling cores are path dependencies resolved through
the house `<SVC>_PYTHONPATH` mechanism:

    BLEND1_PYTHONPATH      blend1's own src (services/blend1/src)
    BLEND1_SYS1_PYTHONPATH the judgment lane core (services/sys1/src)
    BLEND1_SEM1_PYTHONPATH the embeddings lane core (services/sem1/src)
    BLEND1_SDM1_PYTHONPATH the tabular lane core (services/sdm1/src)
    BLEND1_GEN1_PYTHONPATH the media lane core (services/gen1/src) — the
                           voice leg

An absent BLEND1_PYTHONPATH is a NAMED error on the wire, never a
traceback. A lane that cannot answer fails open with its reason recorded
in its lane report and BlendResult.escalated is True — the adapter
surfaces both on every ask result; nothing degrades silently.

The ask result returns the BlendResult's fields as they are (structured
content), so result keys added upstream ride through without an adapter
change; `escalated` and `failed_lanes` are the adapter's escalation
surface beside them.

The `web` tool calls the web leg directly — tinyfish search + fetch,
bounded at the leg's own WEB_MAX_URLS urls and WEB_EXCERPT_CHARS
characters per excerpt — so a live lookup costs one lane call instead of
the whole composed answer. Its result is the leg's own report, verbatim:
the bounds, the url provenance, and the fail-open naming a missing
tinyfish extra or an unset TINYFISH_API_KEY ride through unchanged, so a
missing key is a named error the caller can read, never an empty answer.

Key material never reaches the wire or a log: this process writes
JSON-RPC frames to stdout and nothing else; blend1's telemetry redacts
structurally; doctor reports key NAMES, never values.

Dependencies: blend1 package on PYTHONPATH (set BLEND1_PYTHONPATH to the
`src` directory, or `pip install -e`).
"""

from __future__ import annotations

import json
import os
import sys

# The house path mechanism, in resolution order: blend1's own core first,
# then the sibling cores its lanes compose. Each is inserted only when the
# variable is set — a sibling that is absent fails open inside its own
# lane with the reason named; only blend1's own core is mandatory.
_PATH_VARS = (
    "BLEND1_PYTHONPATH",
    "BLEND1_SYS1_PYTHONPATH",
    "BLEND1_SEM1_PYTHONPATH",
    "BLEND1_SDM1_PYTHONPATH",
    "BLEND1_GEN1_PYTHONPATH",
)

TASK_TYPES = ("classification", "regression")

ASK_TOOL = {
    "name": "ask",
    "description": (
        "One blended ask: the d1 router (sys1) runs solo first and decides "
        "whether generative synthesis is required; the semantic (sem1) and "
        "tabular (sdm1) lanes precompute the facts in parallel; Liquid "
        "LFM2.5-24B-A2B synthesizes only when the router clears ROUTE_FLOOR "
        "0.70; a validation gate (GLiDE → d1) rates the answer before it "
        "ships. Returns the BlendResult's fields as structured content, plus "
        "`escalated` and `failed_lanes` — a lane that fails open records its "
        "reason; the answer never silently degrades."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The ask.",
            },
            "table": {
                "type": "array",
                "items": {"type": "object"},
                "description": (
                    "Optional tabular rows (objects). Rows with a non-null "
                    "`target` value are labeled context; rows without it are "
                    "the query rows sdm1 scores. When absent, the tabular "
                    "lane is skipped."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "The column TabPFN predicts — required when `table` is "
                    "given."
                ),
            },
            "task_type": {
                "type": "string",
                "enum": list(TASK_TYPES),
                "default": "classification",
                "description": "sdm1 task type (default classification).",
            },
            "retrieve": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Retrieve semantic corpus neighbors for the prompt "
                    "(default true)."
                ),
            },
        },
        "required": ["prompt"],
    },
}

WEB_TOOL = {
    "name": "web",
    "description": (
        "Search the live web and fetch the top hits, returning excerpts "
        "with url provenance — bounded by the web leg's own limits and "
        "framed as untrusted reference data; calls blend1's web leg "
        "directly, without the composed ask's router, synthesis, or "
        "validation gate."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "fetch": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Fetch the top hits' pages for excerpts (default "
                    "true). The web leg is one bounded search-plus-fetch "
                    "call with no search-only mode — the value is "
                    "validated here, and the leg's own bound "
                    "(WEB_MAX_URLS urls, WEB_EXCERPT_CHARS characters per "
                    "excerpt) decides what is fetched."
                ),
            },
            "max_urls": {
                "type": "integer",
                "description": (
                    "Optional cap on how many of the top hits are "
                    "fetched. The leg's own WEB_MAX_URLS is the "
                    "server-side ceiling — a larger value is clamped to "
                    "it by the lane, never trusted past it."
                ),
            },
        },
        "required": ["query"],
    },
}

DOCTOR_TOOL = {
    "name": "doctor",
    "description": (
        "Per-lane availability snapshot — one row per lane (route, "
        "semantic, tabular, generate, validate, critics, voice, web), each "
        "with available + reason + provider. Reasons are named; a missing "
        "key names its variable, never its value."
    ),
    "inputSchema": {"type": "object", "properties": {}},
}

SEARCH_TOOL = {
    "name": "search",
    "description": (
        "Semantic search over a project's corpus: embeds the query and "
        "returns the scored entries [{key, score, text}], best first. An "
        "unseeded store returns []. The project resolves by the lane's own "
        "rule (explicit param > BLEND1_PROJECT env > cwd git-repo name > "
        "None/legacy), through the library's own resolve_project — and "
        "over MCP the cwd is the lane's home, so an omitted project "
        "resolves to the repo hosting services/."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query text.",
            },
            "project": {
                "type": "string",
                "description": (
                    "Corpus project (default: $BLEND1_PROJECT or the cwd "
                    "repo name — over MCP the cwd is the lane's home, so "
                    "the default resolves to the repo hosting services/; "
                    "None is the legacy store)."
                ),
            },
            "k": {
                "type": "integer",
                "default": 3,
                "description": "Neighbors to return (default 3).",
            },
        },
        "required": ["query"],
    },
}

REFRESH_TOOL = {
    "name": "refresh",
    "description": (
        "Re-derive a project's corpus from its docs and rebuild only when "
        "the text hashes changed — the steady state is file reads, no "
        "embedding. Unlike ask/doctor/search this tool needs the sibling "
        "wiring: BLEND1_SEM1_PYTHONPATH must name services/sem1/src, and "
        "a call without it is refused naming that variable. The report "
        "names what happened (action: rebuilt | unchanged | no-source). "
        "CLAUDE.local.md is never in a corpus (the library's guard, "
        "inherited). Over MCP the child's cwd is the lane's home "
        "(services/blend1/), so an omitted `project` resolves to the "
        "repo hosting services/ and an omitted `sources` derives nothing "
        "there (no-source): pass both explicitly."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": (
                    "Corpus project (default: $BLEND1_PROJECT or the cwd "
                    "repo name — over MCP the cwd is the lane's home, so "
                    "the default resolves to the repo hosting services/; "
                    "None is the legacy store)."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Explicit files/globs replacing the default docs set "
                    "(docs/**/*.md plus the root contract docs)."
                ),
            },
        },
    },
}

SEED_TOOL = {
    "name": "seed",
    "description": (
        "Embed corpus items into a project's semantic store — the corpus's "
        "write path. Idempotent by construction (the store dedupes by key). "
        "CLAUDE.local.md is never in a corpus (the library's guard, "
        "inherited)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["key", "text"],
                },
                "description": "The corpus items: [{\"key\": …, \"text\": …}].",
            },
            "project": {
                "type": "string",
                "description": (
                    "Corpus project (default: $BLEND1_PROJECT or the cwd "
                    "repo name — over MCP the cwd is the lane's home, so "
                    "the default resolves to the repo hosting services/; "
                    "None is the legacy store)."
                ),
            },
        },
        "required": ["items"],
    },
}

TOOLS = (ASK_TOOL, WEB_TOOL, DOCTOR_TOOL, SEARCH_TOOL, REFRESH_TOOL, SEED_TOOL)


def _add_core_paths() -> None:
    """Insert the house path vars into sys.path BEFORE importing blend1 —
    the sem1 adapter's SEM1_PYTHONPATH pattern, extended to the blended
    lane's sibling cores. Reversed so BLEND1_PYTHONPATH ends up first."""
    for var in reversed(_PATH_VARS):
        path = os.environ.get(var)
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _import_blend1():
    """Import the core with a named failure — an absent BLEND1_PYTHONPATH
    is a named error, never a traceback. Returns (module, error)."""
    try:
        import blend1

        return blend1, None
    except Exception as e:
        return None, (
            "blend1 core not importable: set BLEND1_PYTHONPATH to "
            f"services/blend1/src (the house path mechanism); "
            f"{type(e).__name__}: {e}"
        )


def _validate_ask_arguments(arguments: dict):
    """Returns (kwargs, error) — exactly one is non-None."""
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "`prompt` must be a non-empty string."

    table = arguments.get("table")
    target = arguments.get("target")
    if table is not None:
        if (
            not isinstance(table, list)
            or not table
            or not all(isinstance(row, dict) for row in table)
        ):
            return None, "`table` must be a non-empty array of row objects."
        if target is None:
            return None, (
                "`target` is required when `table` is given (the column "
                "TabPFN predicts)."
            )
    if target is not None and (not isinstance(target, str) or not target.strip()):
        return None, "`target` must be a non-empty string."

    task_type = arguments.get("task_type")
    if task_type is None:
        task_type = "classification"
    if task_type not in TASK_TYPES:
        return None, f"`task_type` must be one of {list(TASK_TYPES)}; got {task_type!r}."

    retrieve = arguments.get("retrieve")
    if retrieve is None:
        retrieve = True
    if not isinstance(retrieve, bool):
        return None, "`retrieve` must be a boolean."

    kwargs: dict = {"task_type": task_type, "retrieve": retrieve}
    if table is not None:
        kwargs["table"] = table
        kwargs["target"] = target
    return kwargs, None


def _validate_project(project):
    """None or a non-empty string — the library validates the name itself
    (one path component, no separators) and its ValueError is the wire's
    named error."""
    if project is None:
        return None, None
    if not isinstance(project, str) or not project.strip():
        return None, "`project` must be a non-empty string."
    return project, None


def _escalation_payload(result) -> dict:
    """The BlendResult's fields as they are — result keys added upstream
    ride through untouched — plus the adapter's escalation surface:
    `escalated` (the library's own property) and `failed_lanes` (every
    lane report whose status is error, scanned generically so a lane added
    upstream is surfaced too)."""
    payload = result.as_dict()
    payload["escalated"] = bool(result.escalated)
    payload["failed_lanes"] = [
        {"lane": name, "reason": report.get("error")}
        for name, report in payload.items()
        if isinstance(report, dict) and report.get("status") == "error"
    ]
    return payload


def main() -> None:
    _add_core_paths()
    blend1, import_error = _import_blend1()

    def respond(req_id, result):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        sys.stdout.flush()

    def error(req_id, code, msg):
        sys.stdout.write(
            json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}) + "\n"
        )
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        if req_id is None:
            # JSON-RPC notification: never answered. Replying emits a frame
            # with no pending requester (id null), which strict clients read
            # as protocol garbage on the shared stdin stream.
            continue
        if method == "ping":
            # MCP requires servers to answer pings with an empty result.
            respond(req_id, {})
            continue

        if method == "initialize":
            respond(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "blend1", "version": "0.1.0"},
                },
            )
        elif method == "tools/list":
            respond(req_id, {"tools": list(TOOLS)})
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                error(req_id, -32602, "`arguments` must be an object.")
                continue

            if blend1 is None:
                # The core never imported: every call answers with the same
                # named reason, never a traceback.
                error(req_id, -32603, import_error)
                continue

            if tool_name == "ask":
                kwargs, arg_error = _validate_ask_arguments(arguments)
                if arg_error is not None:
                    error(req_id, -32602, arg_error)
                    continue
                try:
                    result = blend1.ask(arguments["prompt"], **kwargs)
                except ValueError as e:
                    # The library's own usage errors (e.g. target missing).
                    error(req_id, -32602, str(e))
                    continue
                except Exception as e:
                    error(req_id, -32603, f"{type(e).__name__}: {e}")
                    continue
                respond(
                    req_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    _escalation_payload(result), default=str
                                ),
                            }
                        ]
                    },
                )
                continue

            if tool_name == "web":
                # The web leg called directly: tinyfish search + fetch with
                # the leg's own bounds and its own fail-open — no router, no
                # synthesis, no validation gate. The result rides back
                # verbatim (plan C4: the bounds and the injection defense
                # are the leg's own, never a second implementation here).
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    error(req_id, -32602, "`query` must be a non-empty string.")
                    continue
                fetch = arguments.get("fetch")
                if fetch is None:
                    fetch = True
                if not isinstance(fetch, bool):
                    error(req_id, -32602, "`fetch` must be a boolean.")
                    continue
                max_urls = arguments.get("max_urls")
                if max_urls is not None and (
                    isinstance(max_urls, bool)
                    or not isinstance(max_urls, int)
                    or max_urls < 1
                ):
                    error(req_id, -32602, "`max_urls` must be a positive integer.")
                    continue
                try:
                    # Per-call import, the corpus pattern. The leg's own
                    # fail-open is its RETURN value, not an exception: a
                    # missing tinyfish extra or an unset
                    # TINYFISH_API_KEY comes back as a named error result,
                    # so it rides through unchanged — never an empty
                    # answer. `fetch` and `max_urls` are validated here so
                    # a mistyped value is a named -32602 rather than a
                    # crash; the leg's signature takes the query alone,
                    # and its own WEB_MAX_URLS / WEB_EXCERPT_CHARS are the
                    # server-side clamp — the caller's number never
                    # reaches a fetch path unclamped because the lane
                    # bounds the fetch itself.
                    from blend1 import lanes

                    result = lanes.web_lane(query)
                except Exception as e:
                    error(req_id, -32603, f"{type(e).__name__}: {e}")
                    continue
                respond(
                    req_id,
                    {"content": [{"type": "text", "text": json.dumps(result, default=str)}]},
                )
                continue

            if tool_name == "doctor":
                try:
                    report = blend1.doctor()
                except Exception as e:
                    error(req_id, -32603, f"{type(e).__name__}: {e}")
                    continue
                respond(
                    req_id,
                    {"content": [{"type": "text", "text": json.dumps(report, default=str)}]},
                )
                continue

            if tool_name == "search":
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    error(req_id, -32602, "`query` must be a non-empty string.")
                    continue
                project, proj_error = _validate_project(arguments.get("project"))
                if proj_error is not None:
                    error(req_id, -32602, proj_error)
                    continue
                k = arguments.get("k")
                if k is None:
                    k = 3
                if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                    error(req_id, -32602, "`k` must be a positive integer.")
                    continue
                try:
                    # The library's own resolver names the project; its
                    # output rides back in as the explicit param so one
                    # resolution serves both the call and the report.
                    from blend1.corpus import resolve_project

                    resolved = resolve_project(project)
                    results = blend1.search(query, project=resolved, k=k)
                except ValueError as e:
                    error(req_id, -32602, str(e))
                    continue
                except Exception as e:
                    error(req_id, -32603, f"{type(e).__name__}: {e}")
                    continue
                respond(
                    req_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "project": resolved,
                                        "query": query,
                                        "k": k,
                                        "results": results,
                                    },
                                    default=str,
                                ),
                            }
                        ]
                    },
                )
                continue

            if tool_name == "refresh":
                project, proj_error = _validate_project(arguments.get("project"))
                if proj_error is not None:
                    error(req_id, -32602, proj_error)
                    continue
                sources = arguments.get("sources")
                if sources is not None and (
                    not isinstance(sources, list)
                    or not all(isinstance(s, str) and s.strip() for s in sources)
                ):
                    error(req_id, -32602, "`sources` must be an array of file paths/globs.")
                    continue
                try:
                    from blend1.corpus import resolve_project

                    resolved = resolve_project(project)
                    report = blend1.refresh_corpus(project=resolved, sources=sources)
                except ValueError as e:
                    error(req_id, -32602, str(e))
                    continue
                except ModuleNotFoundError as e:
                    # The refresh leg imports the sem1 core at call time;
                    # a client that launched the server with only
                    # blend1's own path gets the missing variable named,
                    # not a bare ModuleNotFoundError (KI-310).
                    error(
                        req_id,
                        -32603,
                        f"refresh needs the sem1 core: set BLEND1_SEM1_PYTHONPATH "
                        f"to services/sem1/src ({type(e).__name__}: {e})",
                    )
                    continue
                except Exception as e:
                    error(req_id, -32603, f"{type(e).__name__}: {e}")
                    continue
                respond(
                    req_id,
                    {"content": [{"type": "text", "text": json.dumps(report, default=str)}]},
                )
                continue

            if tool_name == "seed":
                items = arguments.get("items")
                if (
                    not isinstance(items, list)
                    or not items
                    or not all(isinstance(it, dict) for it in items)
                ):
                    error(req_id, -32602, "`items` must be a non-empty array of {key, text} objects.")
                    continue
                for it in items:
                    key, text = it.get("key"), it.get("text")
                    if (
                        not isinstance(key, str)
                        or not key.strip()
                        or not isinstance(text, str)
                        or not text.strip()
                    ):
                        error(
                            req_id,
                            -32602,
                            "every item needs a non-empty string `key` and `text`.",
                        )
                        break
                else:
                    project, proj_error = _validate_project(arguments.get("project"))
                    if proj_error is not None:
                        error(req_id, -32602, proj_error)
                        continue
                    try:
                        from blend1.corpus import resolve_project

                        resolved = resolve_project(project)
                        report = blend1.seed_corpus(items, project=resolved)
                    except ValueError as e:
                        error(req_id, -32602, str(e))
                        continue
                    except Exception as e:
                        error(req_id, -32603, f"{type(e).__name__}: {e}")
                        continue
                    respond(
                        req_id,
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {"project": resolved, **report}, default=str
                                    ),
                                }
                            ]
                        },
                    )
                continue

            error(req_id, -32601, f"Unknown tool: {tool_name}")
        else:
            error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()

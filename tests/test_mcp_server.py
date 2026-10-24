"""Tests for blend1's MCP stdio adapter — service/mcp_server.py.

The adapter is fleet's carry (VENDOR.json's fleetAddition: the upstream
ships the library plus the FastAPI service, no MCP server), so its tests
live in this vendored tree's tests/ dir — the same place upstream blend1
keeps its own tests, and outside the re-vendor copy (tests/ is excluded),
so a re-vendor cannot clobber them.

Every test drives the REAL server over stdio the way
service/README.md's hand-test does: JSON-RPC frames in, frames out. The
child runs with a scratch HOME and no TINYFISH_API_KEY, so the unkeyed
state is deterministic — never whatever the developer's ~/.config happens
to hold.

Run: python3 -m pytest services/blend1/tests/test_mcp_server.py
"""

import json
import os
import subprocess

import pytest
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent / "service"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SERVER = SERVICE_DIR / "mcp_server.py"

# A stand-in for the leg's own success report — the shape web_lane
# returns — so "verbatim" has something exact to compare against.
FAKE_LEG_RESULT = {
    "status": "ok",
    "provider": "tinyfish",
    "results": [
        {"url": "https://example.com/a", "title": "A", "text": "excerpt a"},
        {"url": "https://example.com/b", "title": "B", "text": "excerpt b"},
    ],
    "n_results": 2,
    "errors": [],
    "latency_ms": 42,
}

# Boots the real adapter with blend1.lanes.web_lane replaced by a fake
# that echoes the prompt it was called with — proving the handler calls
# the leg with the query and rides the leg's own return value back
# untouched, reshaped by nothing.
FAKE_LEG_WRAPPER = """
import sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {service!r})
import blend1.lanes as lanes

def _fake_web_lane(prompt):
    return dict({fake!r}, seen_prompt=prompt)

lanes.web_lane = _fake_web_lane
import mcp_server
mcp_server.main()
"""


def _child_env(home: Path) -> dict:
    """The child's environment: the house path mechanism
    (BLEND1_PYTHONPATH) plus a scratch HOME so blend1's env loading finds
    no key files. TINYFISH_API_KEY is removed for the same reason — the
    unkeyed state is the test's subject, not the machine's."""
    env = {k: v for k, v in os.environ.items() if k != "TINYFISH_API_KEY"}
    env["BLEND1_PYTHONPATH"] = str(SRC_DIR)
    env["HOME"] = str(home)
    return env


def _drive(frames, tmp_path, wrapper=None):
    """Boot the server (or the fake-leg wrapper) over stdio, feed the
    frames, return the parsed replies in order."""
    if wrapper is None:
        argv = [sys.executable, "-u", str(SERVER)]
    else:
        path = tmp_path / "fake_leg_server.py"
        path.write_text(wrapper, encoding="utf-8")
        argv = [sys.executable, "-u", str(path)]
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    stdin = "".join(json.dumps(f) + "\n" for f in frames)
    proc = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        env=_child_env(home),
        cwd=str(SERVICE_DIR),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def _call(name, arguments, req_id=2):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _result_payload(reply):
    """A tools/call reply's structured content, parsed."""
    assert "result" in reply, reply
    content = reply["result"]["content"]
    assert content and content[0]["type"] == "text"
    return json.loads(content[0]["text"])


def test_tools_list_declares_web_beside_ask(tmp_path):
    replies = _drive([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], tmp_path)
    tools = replies[0]["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "web" in names and "ask" in names
    schema = next(t for t in tools if t["name"] == "web")["inputSchema"]
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "fetch", "max_urls"}
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["fetch"]["type"] == "boolean"
    assert schema["properties"]["fetch"]["default"] is True
    assert schema["properties"]["max_urls"]["type"] == "integer"


def test_web_call_rides_the_legs_own_fail_open_through(tmp_path):
    """Unkeyed: the leg's own named error — the missing tinyfish extra
    here, the unset TINYFISH_API_KEY on a machine where tinyfish is
    installed — is the result, never an empty answer."""
    replies = _drive([_call("web", {"query": "what is the a.test API?"})], tmp_path)
    payload = _result_payload(replies[0])
    assert payload["status"] == "error"
    error = payload["error"]
    assert "tinyfish" in error.lower() or "TINYFISH_API_KEY" in error, error


def test_web_call_returns_the_leg_result_verbatim(tmp_path):
    replies = _drive(
        [_call("web", {"query": "what is the a.test API?"})],
        tmp_path,
        wrapper=FAKE_LEG_WRAPPER.format(
            src=str(SRC_DIR), service=str(SERVICE_DIR), fake=FAKE_LEG_RESULT
        ),
    )
    payload = _result_payload(replies[0])
    # The leg's own report, reshaped by nothing — plus the prompt the
    # handler passed it, echoed by the fake.
    assert payload == dict(FAKE_LEG_RESULT, seen_prompt="what is the a.test API?")


def test_web_rejects_malformed_arguments(tmp_path):
    replies = _drive(
        [
            _call("web", {}, req_id=2),
            _call("web", {"query": "x", "fetch": "yes"}, req_id=3),
            _call("web", {"query": "x", "max_urls": 0}, req_id=4),
        ],
        tmp_path,
    )
    assert len(replies) == 3
    for reply in replies:
        assert reply["error"]["code"] == -32602, reply
    assert "query" in replies[0]["error"]["message"]
    assert "fetch" in replies[1]["error"]["message"]
    assert "max_urls" in replies[2]["error"]["message"]


def test_refresh_without_the_sibling_wiring_names_the_variable(tmp_path):
    """KI-310: refresh imports the sem1 core at call time, so a client
    that launched the server with only blend1's own path gets the missing
    variable NAMED — "refresh needs the sem1 core: set
    BLEND1_SEM1_PYTHONPATH to services/sem1/src" — not a bare
    ModuleNotFoundError wearing a -32603."""
    # The row's precondition, enforced rather than assumed: the child's
    # interpreter must not carry sem1 (a project venv that installed it
    # makes the refusal unreachable and a plain refresh legitimate —
    # skip there and name why).
    probe = subprocess.run(
        [sys.executable, "-c", "import sem1"],
        capture_output=True,
        env={k: v for k, v in _child_env(tmp_path).items() if k != "BLEND1_PYTHONPATH"},
    )
    if probe.returncode == 0:
        pytest.skip("the child's interpreter imports sem1; no missing variable to name")
    # A source that exists, so the refresh reaches the embed leg — the
    # only leg that imports the sem1 core (a no-source refresh returns
    # before the import and would answer success).
    source = tmp_path / "doc.md"
    source.write_text("## Notes\n\nbody\n", encoding="utf-8")
    replies = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            _call(
                "refresh",
                {"project": "some-project", "sources": [str(source)]},
            ),
        ],
        tmp_path,
    )
    reply = replies[-1]
    assert "error" in reply, reply
    assert reply["error"]["code"] == -32603, reply
    message = reply["error"]["message"]
    assert "refresh needs the sem1 core" in message, message
    assert "BLEND1_SEM1_PYTHONPATH" in message, message
    assert "services/sem1/src" in message, message

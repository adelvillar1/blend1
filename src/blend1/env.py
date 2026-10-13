"""Lane key/env loading — the house key-file pattern.

Resolution order: real environment wins (setdefault semantics) →
~/.config/blend1/env (canonical, chmod 600, never committed) →
~/.config/sdm1/env (the tabular lane's key home) →
~/.config/dev-decisions/env (legacy fallback — where the Liquid keys
already live). Every file that exists is loaded, in that order, so the
blend and the sibling lanes never disagree about which key is loaded.

Names are reported, never values. A missing file is a no-op; a missing key
surfaces later as the lane's own fail-open error naming the variable (never
a silent degradation).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

ENV_FILE = Path.home() / ".config" / "blend1" / "env"

# Keys the lanes read from the environment:
#   LIQUID_API_KEY   — BOTH hosted Liquid lanes: the d1 router (sys1 reads it
#                      from its own env file) and the LFM2.5-24B generative
#                      leg (blend1 reads it here). One vendor, one key.
#   TABPFN_API_KEY   — tabular lane (Prior Labs cloud API; read by sdm1)
#   BLEND1_SEMANTIC_ENDPOINT — semantic lane (default http://127.0.0.1:8901)
#   DASHSCOPE_API_KEY / STEPFUN_API_KEY — voice lane (read by gen1)


def _parse_lines(text: str, out: dict) -> None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, _, val = line.partition("=")
        name = name.strip()
        if not name or not name.replace("_", "a").isalnum():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if name not in os.environ:
            os.environ[name] = val
            out["loaded"].append(name)


def load_env(path: Path | None = None) -> dict:
    """Load env file(s) into os.environ (setdefault). Returns {path, loaded}
    — key NAMES only, never values (same line the rest of the codebase holds).
    """
    if path is not None:
        paths = [path]
    else:
        paths = [ENV_FILE]
        for extra in (
            Path.home() / ".config" / "sdm1" / "env",
            Path.home() / ".config" / "dev-decisions" / "env",
            Path.home() / ".config" / "gen1" / "env",
            Path.home() / ".hermes" / ".env",
        ):
            if extra not in paths:
                paths.append(extra)
    out: dict = {"path": None, "loaded": []}
    for p in paths:
        if not p.exists():
            continue
        if out["path"] is None:
            out["path"] = str(p)
        with contextlib.suppress(Exception):
            _parse_lines(p.read_text(encoding="utf-8"), out)
    return out

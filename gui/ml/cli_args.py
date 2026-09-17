from __future__ import annotations

import functools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

_OPT_RE = re.compile(r"(?<![\w-])--[A-Za-z0-9][\w-]*")


@functools.lru_cache(maxsize=32)
def supported_options(script: str, timeout: float = 120.0) -> frozenset[str]:
    """Return long options advertised by a script's --help output."""
    try:
        path = Path(script).resolve()
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(path), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(path.parent),
        )
        return frozenset(_OPT_RE.findall((result.stdout or "") + (result.stderr or "")))
    except Exception:
        return frozenset()


def build_argv(script: str, options: Mapping[str, Any], *, log=print) -> list[str]:
    """Build argv from options, retaining only options supported by the script."""
    known = supported_options(script)
    argv: list[str] = []
    for key, value in options.items():
        flag = key if key.startswith("--") else "--" + key.replace("_", "-")
        if value is None or value is False or value == "":
            continue
        if known and flag not in known:
            log(f"опция {flag} не поддерживается {Path(script).name} — пропущена")
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple, set)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend((flag, str(value)))
    return argv

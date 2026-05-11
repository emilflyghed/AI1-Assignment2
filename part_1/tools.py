"""Single tool: bash command executor via subprocess.run(shell=True).

Combined stdout+stderr returned as a string. Truncated to ~MAX_OUTPUT chars with
a clear marker. Timeout and non-zero exit are reported as text and never raised.
"""
from __future__ import annotations

import subprocess

TIMEOUT = 30
MAX_OUTPUT = 2000
TRUNC_MARKER = "\n... [output truncated] ..."


def run_bash(command: str) -> str:
    if not command or not command.strip():
        return "[error] empty command"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {TIMEOUT}s"
    except Exception as e:  # surface unexpected failures rather than crash the loop
        return f"[error] {type(e).__name__}: {e}"

    out = result.stdout or ""
    err = result.stderr or ""

    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(("[stderr]\n" if out else "") + err)
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")

    combined = "\n".join(parts) if parts else ""
    return _truncate(combined)


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT:
        return s
    return s[:MAX_OUTPUT] + TRUNC_MARKER

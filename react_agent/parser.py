"""Parse raw model text into Thought / Action / Action Input / Final Answer.

Robust to extra whitespace and minor formatting drift. Case-insensitive labels.
If both `Action:` and `Final Answer:` appear, Final Answer wins.
On malformed output, returns ParseResult(kind="error") so the loop can re-prompt.
Action Input must be one physical line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

LABELS = ("Thought", "Action", "Action Input", "Observation", "Final Answer")
_LABEL_ALT = "|".join(re.escape(l) for l in LABELS)


@dataclass
class ParseResult:
    kind: str  # "action" | "final" | "error"
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None


def _extract_block(label: str, text: str) -> Optional[str]:
    """Capture a multi-line section: from `Label:` until the next known label or EOF."""
    pattern = (
        rf'(?:^|\n)\s*{re.escape(label)}\s*:\s*(.*?)'
        rf'(?=\n\s*(?:{_LABEL_ALT})\s*:|\Z)'
    )
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_line(label: str, text: str) -> Optional[str]:
    """Capture only the rest of the line after `Label:`. Used for Action."""
    pattern = rf'(?:^|\n)\s*{re.escape(label)}\s*:\s*(.+?)(?=\n|$)'
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse(text: str) -> ParseResult:
    if text is None or not text.strip():
        return ParseResult(kind="error", error="empty response")

    thought = _extract_block("Thought", text)
    final_answer = _extract_block("Final Answer", text)

    if final_answer is not None and final_answer != "":
        return ParseResult(kind="final", thought=thought, final_answer=final_answer)

    action = _extract_line("Action", text)
    action_input = _extract_block("Action Input", text)

    if action and action_input:
        if "\n" in action_input or "\r" in action_input:
            return ParseResult(
                kind="error",
                error="Action Input must be a single line",
            )
        return ParseResult(
            kind="action",
            thought=thought,
            action=action,
            action_input=action_input,
        )

    if action and not action_input:
        return ParseResult(
            kind="error",
            error="Action found but Action Input is missing",
        )
    if action_input and not action:
        return ParseResult(
            kind="error",
            error="Action Input found but Action is missing",
        )

    return ParseResult(
        kind="error",
        error="no Action or Final Answer found",
    )

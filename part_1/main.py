"""CLI entrypoint. Runs the ReAct loop end-to-end with a MockLLM and a hard-coded scenario.

Usage:
    python main.py                     # runs the demo scenario
    python main.py "your task here"    # runs the demo scenario but prints the given task

The MockLLM script is fixed, so the demo proves the loop works without any API key.
"""
from __future__ import annotations

import sys

from agent import run
from llm import MockLLM


# Hard-coded scenario: count Python files in the current directory.
DEMO_TASK = "How many Python files are in this directory?"

DEMO_SCRIPT = [
    # Turn 1: model picks an action.
    (
        "Thought: I need to count Python files in this directory tree. "
        "I'll use find piped to wc -l to get the count.\n"
        "Action: bash\n"
        "Action Input: find . -name \"*.py\" | wc -l"
    ),
    # Turn 2: model reads the observation and answers.
    (
        "Thought: The previous bash command returned a count. I have what I need.\n"
        "Final Answer: According to the find/wc output above, that number is the "
        "count of Python files in the current directory tree."
    ),
]


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else DEMO_TASK
    llm = MockLLM(DEMO_SCRIPT)
    answer = run(llm, task)
    print("===== Final Answer =====")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

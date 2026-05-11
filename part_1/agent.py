"""Main ReAct loop.

Sends system prompt + user task + accumulated history to the LLM, parses the
response, executes bash if asked, appends the observation, and repeats. Caps
iterations at MAX_ITERS to prevent runaways.
"""
from __future__ import annotations

from parser import parse
from prompt import SYSTEM_PROMPT
from tools import run_bash

MAX_ITERS = 10


def _build_prompt(user_task: str, history: str, correction: str | None) -> str:
    parts = [SYSTEM_PROMPT, "", f"User task: {user_task}", ""]
    if history:
        parts.append(history.rstrip())
        parts.append("")
    if correction:
        parts.append(f"[system note] {correction}")
        parts.append("")
    return "\n".join(parts)


def run(llm, user_task: str) -> str:
    """Run the ReAct loop until Final Answer or iteration cap. Returns the answer string."""
    history = ""
    correction: str | None = None

    for _ in range(MAX_ITERS):
        prompt = _build_prompt(user_task, history, correction)
        correction = None

        response = llm.complete(prompt)
        result = parse(response)

        if result.kind == "final":
            return result.final_answer or ""

        if result.kind == "error":
            # Re-prompt the model with a correction note next turn.
            correction = (
                f"Your previous response could not be parsed: {result.error}. "
                "Please respond strictly in the required format "
                "(Thought + Action + Action Input, OR Thought + Final Answer)."
            )
            history += response.rstrip() + "\n"
            continue

        # action
        observation = run_bash(result.action_input or "")
        history += response.rstrip() + "\n" + f"Observation: {observation}\n"

    return f"[agent stopped: reached max iterations ({MAX_ITERS}) without Final Answer]"

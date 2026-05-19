"""CLI entrypoint for the ReAct agent.

Usage:
    python main.py                     # runs the default task with Gemini
    python main.py --provider lmstudio # runs the default task with LM Studio
    python main.py --verbose           # prints model turns and observations
    python main.py --mock              # runs the offline demo scenario

Config can come from environment variables or the project .env file.
"""
from __future__ import annotations

import argparse
import sys

from agent import run
from llm import (
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_MODEL,
    LMStudioLLM,
    MockLLM,
    RealLLM,
    load_config_value,
)


DEFAULT_TASK = (
    "How many Python files are in this repository tree? Search recursively from "
    "the current working directory and ignore __pycache__ directories."
)
PROVIDER_ENV_VAR = "LLM_PROVIDER"
DEFAULT_PROVIDER = "gemini"

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


def _default_provider() -> str:
    provider = (load_config_value(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER).strip().lower()
    if provider not in {"gemini", "lmstudio"}:
        raise ValueError(
            f"Unsupported {PROVIDER_ENV_VAR}={provider!r}. "
            "Use 'gemini' or 'lmstudio'."
        )
    return provider


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Part 1 ReAct agent.")
    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent. If omitted, a small default task is used.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the fixed offline demo instead of calling the real LLM API.",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "lmstudio"),
        default=None,
        help="LLM provider to use. Can also be set with LLM_PROVIDER.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Gemini defaults to "
            f"{DEFAULT_MODEL}; LM Studio defaults to LM_STUDIO_MODEL or the "
            "first loaded local model."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "LM Studio OpenAI-compatible base URL. Defaults to "
            f"{DEFAULT_LM_STUDIO_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each model response and tool observation before the final answer.",
    )
    args = parser.parse_args(argv)
    if args.provider is None:
        args.provider = _default_provider()
    return args


def _task_from_parts(parts: list[str]) -> str:
    task = " ".join(parts).strip()
    return task or DEFAULT_TASK


def _make_llm(args: argparse.Namespace):
    if args.mock:
        return MockLLM(DEMO_SCRIPT)
    if args.provider == "lmstudio":
        return LMStudioLLM(model=args.model, base_url=args.base_url)
    return RealLLM(model=args.model or DEFAULT_MODEL)


def _print_trace(kind: str, text: str) -> None:
    print(f"===== {kind.title()} =====")
    print(text if text else "(empty)")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    task = _task_from_parts(args.task)
    llm = _make_llm(args)
    answer = run(llm, task, on_step=_print_trace if args.verbose else None)
    print("===== Final Answer =====")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

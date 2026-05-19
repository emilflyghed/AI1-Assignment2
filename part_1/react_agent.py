#!/usr/bin/env python3
"""Part 1 ReAct agent in one Python file.

This is intentionally small and explicit:
- no agent frameworks
- no provider-native tool/function calling
- raw text parsing for a homemade BASH tool call
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib import error, request


MAX_ITERATIONS = 8
BASH_TIMEOUT_SECONDS = 10
MAX_TOOL_OUTPUT_CHARS = 2000

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"

SYSTEM_PROMPT = f"""
You are a helpful coding assistant that can execute bash commands.

When you need to run a bash command, output EXACTLY this format on its own line:
BASH: <command here>

Examples:
BASH: ls -la
BASH: python3 --version
BASH: cat README.md

Rules:
- Only output ONE BASH: line per response.
- After seeing the command output, reason about what to do next.
- When you have enough information to answer the user, respond in plain text WITHOUT a BASH line.
- Never run destructive commands (rm, mv to dangerous paths, etc.).
- If a command output is very long, focus on the relevant parts.
- Command output is limited to {MAX_TOOL_OUTPUT_CHARS} characters.
- Maximum command rounds: {MAX_ITERATIONS}.
""".strip()

BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\b.*\bof=",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\s+-R\b",
    r"\bkill\s+-9\s+1\b",
    r">\s*/dev/sd[a-z]",
    r":\(\)\s*\{",
]


def read_dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        if name.strip() != key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    return None


def config_value(*keys: str, default: str | None = None) -> str | None:
    """Read config from env, part_1/.env, or assignment-root .env."""
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value

    part_root = Path(__file__).resolve().parent
    for dotenv_path in (part_root / ".env", part_root.parent / ".env"):
        for key in keys:
            value = read_dotenv_value(dotenv_path, key)
            if value:
                return value

    return default


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str | None,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint with plain messages."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    api_request = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(api_request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    data = json.loads(response_text)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data}") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM response did not contain text: {data}")
    return content.strip()


def extract_bash_command(reply: str) -> str | None:
    """Parse the homemade tool call from raw text."""
    for line in reply.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("BASH:"):
            return stripped.split(":", 1)[1].strip()

    return None


def is_command_safe(command: str) -> bool:
    if not command.strip():
        return False
    if "\n" in command or "\r" in command:
        return False
    if len(command) > 500:
        return False

    lowered = command.lower()
    return not any(re.search(pattern, lowered) for pattern in BLOCKED_COMMAND_PATTERNS)


def execute_bash(command: str) -> str:
    if not is_command_safe(command):
        return "[BLOCKED: Command rejected by safety filter]"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT: Command exceeded {BASH_TIMEOUT_SECONDS} seconds]"

    output = (result.stdout or "") + (result.stderr or "")
    if not output:
        output = "(no output)"
    if result.returncode != 0:
        output = f"[exit code {result.returncode}]\n{output}"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n[TRUNCATED]"
    return output


def react_loop(
    user_question: str,
    complete: Callable[[list[dict[str, str]]], str],
) -> str:
    """Main ReAct loop: Reason -> Act -> Observe -> Repeat."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        reply = complete(messages)
        bash_command = extract_bash_command(reply)

        if bash_command is None:
            print(f"[{iteration}/{MAX_ITERATIONS}] Final answer ready.")
            return reply

        print(f"[{iteration}/{MAX_ITERATIONS}] BASH: {bash_command}")
        observation = execute_bash(bash_command)
        preview = observation[:100].replace("\n", " ")
        print(f"[Output: {preview}...]")

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Command output:\n{observation}"})

    return "[Agent stopped: iteration limit reached]"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-file Part 1 ReAct agent.")
    parser.add_argument("question", nargs="*", help="Question/task for the agent.")
    parser.add_argument("--model", default=None, help=f"Model name. Default: {DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help=f"LM Studio base URL. Default: {DEFAULT_BASE_URL}.")
    parser.add_argument("--api-key", default=None, help="API key for the LLM provider.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    user_question = " ".join(args.question).strip() or "Count Python files in this folder."

    model = args.model or config_value(
        "LM_STUDIO_MODEL",
        "OPENAI_MODEL",
        "OPENROUTER_MODEL",
        "LLM_MODEL",
        default=DEFAULT_MODEL,
    )
    base_url = args.base_url or config_value(
        "LM_STUDIO_BASE_URL",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        default=DEFAULT_BASE_URL,
    )
    api_key = args.api_key or config_value("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_API")

    complete = lambda messages: chat_completion(
        messages=messages,
        model=model or DEFAULT_MODEL,
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
    )

    answer = react_loop(user_question, complete)
    print("\n===== Final Answer =====")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

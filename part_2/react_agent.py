#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib import error, request


MAX_TOOL_ROUNDS = 8
BASH_TIMEOUT_SECONDS = 10
MAX_TOOL_OUTPUT_CHARS = 2000
MAX_EDIT_FILE_BYTES = 200_000
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"
DEFAULT_SYSTEM_PROMPT_PATH = "system_prompt.txt"

BLOCKED_COMMAND_PATTERNS = [
    r"\brm\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bmkdir\b",
    r"\btouch\b",
    r"\btee\b",
    r"\btruncate\b",
    r"\bsed\s+-i\b",
    r"\bkill\s+-9\s+1\b",
    r">\s*/dev/sd[a-z]",
    r":\(\)\s*\{",
]

BLOCKED_SHELL_TOKENS = [";", "&&", "||", "`", "$(", ">", ">>", "<"]


class ModelOutputError(ValueError):
    """Raised when the model does not return the required structured output."""


class LLMHTTPError(RuntimeError):
    """HTTP error from the LLM endpoint."""

    def __init__(self, code: int, details: str) -> None:
        super().__init__(f"LLM request failed with HTTP {code}: {details}")
        self.code = code
        self.details = details


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
    """Read config from env, part_2/.env, or assignment-root .env."""
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


def load_system_prompt(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.read_text(encoding="utf-8").strip()


def _post_chat_json(
    payload: dict[str, Any],
    base_url: str,
    api_key: str | None,
) -> dict[str, Any]:
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
        raise LLMHTTPError(exc.code, details) from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON HTTP response: {response_text}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"LLM returned unexpected HTTP response: {data}")
    return data


def chat_completion(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str | None,
) -> str:
    """Call an OpenAI-compatible endpoint and request a JSON object response."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        data = _post_chat_json(payload, base_url, api_key)
    except LLMHTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        payload.pop("response_format", None)
        data = _post_chat_json(payload, base_url, api_key)

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {data}") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM response did not contain text: {data}")
    return content.strip()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_model_action(reply: str) -> dict[str, Any]:
    """Parse and validate the model's structured JSON response."""
    candidate = strip_json_fence(reply)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ModelOutputError("response was not valid JSON") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ModelOutputError("response JSON must be an object")

    action = parsed.get("action")
    if action == "final":
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ModelOutputError("final response must include non-empty string field 'answer'")
        return {"action": "final", "answer": answer.strip()}

    if action == "tool":
        tool = parsed.get("tool")
        args = parsed.get("args")
        if not isinstance(tool, str) or not tool:
            raise ModelOutputError("tool response must include string field 'tool'")
        if not isinstance(args, dict):
            raise ModelOutputError("tool response must include object field 'args'")
        return {"action": "tool", "tool": tool, "args": args}

    raise ModelOutputError("response action must be 'tool' or 'final'")


def truncate_tool_output(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_TOOL_OUTPUT_CHARS]
        + f"\n[TRUNCATED: tool output limited to {MAX_TOOL_OUTPUT_CHARS} characters]"
    )


def is_command_safe(command: str) -> tuple[bool, str]:
    if not command.strip():
        return False, "empty command"
    if "\n" in command or "\r" in command:
        return False, "multi-line commands are not allowed"
    if len(command) > 500:
        return False, "command is too long"

    lowered = command.lower()
    for token in BLOCKED_SHELL_TOKENS:
        if token in command:
            return False, f"blocked shell token: {token}"

    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"blocked command pattern: {pattern}"

    return True, "ok"


def execute_bash(args: dict[str, Any]) -> str:
    command = args.get("command")
    if not isinstance(command, str):
        return "[ERROR: bash requires args.command as a string]"

    safe, reason = is_command_safe(command)
    if not safe:
        return f"[BLOCKED: Command rejected by safety filter: {reason}]"

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
    return truncate_tool_output(output)


def safe_workspace_path(path_text: Any) -> Path:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("path must be a non-empty string")

    raw_path = Path(path_text)
    if raw_path.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in raw_path.parts:
        raise ValueError("parent-directory paths are not allowed")

    workspace = Path.cwd().resolve()
    path = (workspace / raw_path).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("path escapes the workspace")
    return path


def read_editable_text(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"file does not exist: {path.relative_to(Path.cwd())}")
    if not path.is_file():
        raise ValueError("path must point to a file")
    if path.stat().st_size > MAX_EDIT_FILE_BYTES:
        raise ValueError(f"file is too large to edit; limit is {MAX_EDIT_FILE_BYTES} bytes")

    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary files are not editable")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("file must be UTF-8 text") from exc


def replace_line_range(text: str, start_line: Any, end_line: Any, new_text: str) -> tuple[str, str]:
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise ValueError("start_line and end_line must be integers")
    lines = text.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line or end_line > len(lines):
        raise ValueError("line range is outside the file")

    replacement = new_text.splitlines(keepends=True)
    if new_text and not new_text.endswith("\n"):
        replacement.append("\n")

    updated = lines[: start_line - 1] + replacement + lines[end_line:]
    return "".join(updated), f"lines {start_line}-{end_line}"


def replace_marker_range(
    text: str,
    start_marker: Any,
    end_marker: Any,
    new_text: str,
    include_markers: bool,
) -> tuple[str, str]:
    if not isinstance(start_marker, str) or not isinstance(end_marker, str):
        raise ValueError("start_marker and end_marker must be strings")

    lines = text.splitlines(keepends=True)
    start_index = next((i for i, line in enumerate(lines) if start_marker in line), None)
    end_index = next((i for i, line in enumerate(lines) if end_marker in line), None)
    if start_index is None or end_index is None or end_index <= start_index:
        raise ValueError("marker range was not found")

    replace_start = start_index if include_markers else start_index + 1
    replace_end = end_index + 1 if include_markers else end_index

    replacement = new_text.splitlines(keepends=True)
    if new_text and not new_text.endswith("\n"):
        replacement.append("\n")

    updated = lines[:replace_start] + replacement + lines[replace_end:]
    return "".join(updated), f"marker range {start_marker!r} to {end_marker!r}"


def edit_file_section(args: dict[str, Any]) -> str:
    try:
        path = safe_workspace_path(args.get("path"))
        new_text = args.get("new_text")
        if not isinstance(new_text, str):
            raise ValueError("new_text must be a string")

        original = read_editable_text(path)
        if "start_line" in args or "end_line" in args:
            updated, description = replace_line_range(
                original,
                args.get("start_line"),
                args.get("end_line"),
                new_text,
            )
        else:
            updated, description = replace_marker_range(
                original,
                args.get("start_marker"),
                args.get("end_marker"),
                new_text,
                bool(args.get("include_markers", False)),
            )

        path.write_text(updated, encoding="utf-8")
        relative = path.relative_to(Path.cwd())
        return f"[OK: edited {relative} at {description}]"
    except OSError as exc:
        return f"[ERROR: edit_file_section failed: {exc}]"
    except ValueError as exc:
        return f"[ERROR: edit_file_section rejected request: {exc}]"


def dispatch_tool(tool: str, args: dict[str, Any]) -> str:
    if tool == "bash":
        return execute_bash(args)
    if tool == "edit_file_section":
        return edit_file_section(args)
    return f"[ERROR: unknown tool {tool!r}]"


def tool_result_message(tool: str, result: str) -> str:
    return (
        f"TOOL_RESULT for {tool}:\n"
        f"{result}\n\n"
        "Continue with another JSON tool call if more work is needed, or return "
        "a JSON final answer if the task is complete."
    )


def run_agent(
    messages: list[dict[str, str]],
    complete: Callable[[list[dict[str, str]]], str],
) -> str:
    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        reply = complete(messages)

        try:
            action = parse_model_action(reply)
        except ModelOutputError as exc:
            print(f"[{round_number}/{MAX_TOOL_ROUNDS}] Invalid JSON response: {exc}")
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required JSON schema. "
                        "Return only one JSON object with action='tool' or action='final'."
                    ),
                }
            )
            continue

        messages.append({"role": "assistant", "content": reply})

        if action["action"] == "final":
            print(f"[{round_number}/{MAX_TOOL_ROUNDS}] Final answer ready.")
            return action["answer"]

        tool = action["tool"]
        args = action["args"]
        print(f"[{round_number}/{MAX_TOOL_ROUNDS}] TOOL: {tool}")
        result = dispatch_tool(tool, args)
        preview = result[:100].replace("\n", " ")
        print(f"[Output: {preview}...]")
        messages.append({"role": "user", "content": tool_result_message(tool, result)})

    return "[Agent stopped: tool-round limit reached]"


def make_initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-file Part 2 structured agent.")
    parser.add_argument("question", nargs="*", help="Question/task for the agent.")
    parser.add_argument("--model", default=None, help=f"Model name. Default: {DEFAULT_MODEL}.")
    parser.add_argument("--base-url", default=None, help=f"LM Studio base URL. Default: {DEFAULT_BASE_URL}.")
    parser.add_argument("--api-key", default=None, help="API key for the LLM provider.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help=f"Path to the system-prompt config file. Default: {DEFAULT_SYSTEM_PROMPT_PATH}.",
    )
    return parser.parse_args(argv)


def build_complete(args: argparse.Namespace) -> Callable[[list[dict[str, str]]], str]:
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

    return lambda messages: chat_completion(
        messages=messages,
        model=model or DEFAULT_MODEL,
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
    )


def run_one_shot(question: str, system_prompt: str, complete: Callable[[list[dict[str, str]]], str]) -> str:
    messages = make_initial_messages(system_prompt)
    messages.append({"role": "user", "content": question})
    return run_agent(messages, complete)


def run_interactive(system_prompt: str, complete: Callable[[list[dict[str, str]]], str]) -> None:
    messages = make_initial_messages(system_prompt)
    print("Part 2 agent session. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nuser> ").strip()
        except EOFError:
            print()
            return
        if user_input.lower() in {"exit", "quit"}:
            return
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        answer = run_agent(messages, complete)
        print("\n===== Final Answer =====")
        print(answer)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    system_prompt = load_system_prompt(args.system_prompt)
    complete = build_complete(args)
    question = " ".join(args.question).strip()

    if question:
        answer = run_one_shot(question, system_prompt, complete)
        print("\n===== Final Answer =====")
        print(answer)
    else:
        run_interactive(system_prompt, complete)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

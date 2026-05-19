"""LLM client wrappers for the ReAct agent."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List
from urllib import error, parse, request

DEFAULT_MODEL = "gemini-flash-lite-latest"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
API_KEY_ENV_VAR = "LLM_API"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_BASE_URL_ENV_VAR = "LM_STUDIO_BASE_URL"
LM_STUDIO_MODEL_ENV_VAR = "LM_STUDIO_MODEL"


def _read_dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        if name.strip() != key:
            continue

        stripped_value = value.strip()
        if (
            len(stripped_value) >= 2
            and stripped_value[0] == stripped_value[-1]
            and stripped_value[0] in {"'", '"'}
        ):
            return stripped_value[1:-1]
        return stripped_value

    return None


def _dotenv_paths() -> tuple[Path, ...]:
    part_root = Path(__file__).resolve().parent.parent
    return (part_root / ".env", part_root.parent / ".env")


def _load_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        return api_key

    for dotenv_path in _dotenv_paths():
        dotenv_api_key = _read_dotenv_value(dotenv_path, API_KEY_ENV_VAR)
        if dotenv_api_key:
            return dotenv_api_key

    dotenv_locations = ", ".join(str(path) for path in _dotenv_paths())

    raise RuntimeError(
        f"Missing API key. Set {API_KEY_ENV_VAR} in the environment or in "
        f"one of: {dotenv_locations}."
    )


def load_config_value(key: str) -> str | None:
    """Read config from the environment, part .env, or assignment-root .env."""
    value = os.environ.get(key)
    if value:
        return value

    for dotenv_path in _dotenv_paths():
        dotenv_value = _read_dotenv_value(dotenv_path, key)
        if dotenv_value:
            return dotenv_value
    return None


def _post_json_url(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    error_prefix: str,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} failed with HTTP {exc.code}: {error_text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{error_prefix} failed: {exc.reason}") from exc

    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_prefix} returned invalid JSON.") from exc

    if not isinstance(parsed_response, dict):
        raise RuntimeError(f"{error_prefix} returned an unexpected response type.")
    return parsed_response


def _get_json_url(url: str, timeout_seconds: float, error_prefix: str) -> dict[str, Any]:
    api_request = request.Request(url, headers={"Accept": "application/json"}, method="GET")

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} failed with HTTP {exc.code}: {error_text}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{error_prefix} failed: {exc.reason}") from exc

    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_prefix} returned invalid JSON.") from exc

    if not isinstance(parsed_response, dict):
        raise RuntimeError(f"{error_prefix} returned an unexpected response type.")
    return parsed_response


class MockLLM:
    """Returns scripted responses one at a time. For testing the loop offline."""

    def __init__(self, responses: Iterable[str]) -> None:
        self.responses: List[str] = list(responses)
        self._index: int = 0

    def complete(self, prompt: str) -> str:
        if self._index >= len(self.responses):
            raise IndexError(
                f"MockLLM script exhausted after {self._index} responses; "
                "the loop asked for more turns than were scripted."
            )
        response = self.responses[self._index]
        self._index += 1
        return response


class RealLLM:
    """Calls Gemini's plain text generation API using the same interface as MockLLM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")

        self.model = model
        self.api_key = api_key if api_key is not None else _load_api_key()
        if not self.api_key:
            raise ValueError("api_key must not be empty.")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty.")

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        response_body = self._post_json(payload)
        return self._extract_text(response_body)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_model = parse.quote(self.model, safe="")
        encoded_key = parse.quote(self.api_key, safe="")
        url = f"{GEMINI_API_URL}/{encoded_model}:generateContent?key={encoded_key}"
        return _post_json_url(url, payload, self.timeout_seconds, "LLM API request")

    @staticmethod
    def _extract_text(response_body: dict[str, Any]) -> str:
        candidates = response_body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"LLM API response has no candidates: {response_body}")

        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            raise RuntimeError("LLM API candidate has an unexpected response type.")

        content = first_candidate.get("content")
        if not isinstance(content, dict):
            raise RuntimeError(f"LLM API candidate has no content: {first_candidate}")

        parts = content.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError(f"LLM API content has no parts: {content}")

        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if not text_parts:
            raise RuntimeError(f"LLM API response contains no text: {response_body}")

        return "".join(text_parts).strip()


class LMStudioLLM:
    """Calls LM Studio's local OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")

        self.base_url = (
            base_url
            or load_config_value(LM_STUDIO_BASE_URL_ENV_VAR)
            or DEFAULT_LM_STUDIO_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model = model or load_config_value(LM_STUDIO_MODEL_ENV_VAR) or self._first_model_id()
        if not self.model:
            raise ValueError("model must not be empty.")

    def complete(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty.")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        response_body = self._post_json(payload)
        return self._extract_text(response_body)

    def _first_model_id(self) -> str:
        response_body = self._get_json(f"{self.base_url}/models")
        models = response_body.get("data")
        if not isinstance(models, list) or not models:
            raise RuntimeError(
                "LM Studio returned no models. Load Gemma in LM Studio or set "
                f"{LM_STUDIO_MODEL_ENV_VAR} to the exact model identifier."
            )

        first_model = models[0]
        if not isinstance(first_model, dict) or not isinstance(first_model.get("id"), str):
            raise RuntimeError(f"LM Studio returned an unexpected models response: {response_body}")
        return first_model["id"]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json_url(
            f"{self.base_url}/chat/completions",
            payload,
            self.timeout_seconds,
            "LM Studio API request",
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        return _get_json_url(url, self.timeout_seconds, "LM Studio API request")

    @staticmethod
    def _extract_text(response_body: dict[str, Any]) -> str:
        choices = response_body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"LM Studio response has no choices: {response_body}")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("LM Studio choice has an unexpected response type.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(f"LM Studio choice has no message: {first_choice}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"LM Studio response contains no text: {response_body}")
        return content.strip()

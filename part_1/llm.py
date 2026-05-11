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


def _load_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        return api_key

    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    dotenv_api_key = _read_dotenv_value(dotenv_path, API_KEY_ENV_VAR)
    if dotenv_api_key:
        return dotenv_api_key

    raise RuntimeError(
        f"Missing API key. Set {API_KEY_ENV_VAR} in the environment or in "
        f"{dotenv_path}."
    )


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
        body = json.dumps(payload).encode("utf-8")
        api_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(api_request, timeout=self.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM API request failed with HTTP {exc.code}: {error_text}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM API returned invalid JSON.") from exc

        if not isinstance(parsed_response, dict):
            raise RuntimeError("LLM API returned an unexpected response type.")
        return parsed_response

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

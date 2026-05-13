from __future__ import annotations

from typing import Any

import pytest

from llm import LMStudioLLM, RealLLM


class CapturingLLM(RealLLM):
    def __init__(self) -> None:
        super().__init__(api_key="test-key")
        self.payload: dict[str, Any] | None = None

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Thought: ok\nFinal Answer: done"}],
                    },
                },
            ],
        }


def test_real_llm_posts_plain_prompt_and_extracts_text() -> None:
    llm = CapturingLLM()

    response = llm.complete("Solve this task.")

    assert response == "Thought: ok\nFinal Answer: done"
    assert llm.payload is not None
    assert llm.payload["contents"][0]["parts"][0]["text"] == "Solve this task."
    assert "tools" not in llm.payload


def test_real_llm_raises_when_response_has_no_text() -> None:
    with pytest.raises(RuntimeError, match="no text"):
        RealLLM._extract_text({"candidates": [{"content": {"parts": []}}]})


class CapturingLMStudioLLM(LMStudioLLM):
    def __init__(self, model: str | None = "local-model") -> None:
        super().__init__(model=model, base_url="http://localhost:1234/v1")
        self.payload: dict[str, Any] | None = None

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": "Thought: ok\nFinal Answer: local done",
                    },
                },
            ],
        }

    def _get_json(self, url: str) -> dict[str, Any]:
        return {"data": [{"id": "auto-local-model"}]}


def test_lmstudio_llm_posts_openai_compatible_plain_prompt() -> None:
    llm = CapturingLMStudioLLM()

    response = llm.complete("Solve locally.")

    assert response == "Thought: ok\nFinal Answer: local done"
    assert llm.payload is not None
    assert llm.payload["model"] == "local-model"
    assert llm.payload["messages"] == [{"role": "user", "content": "Solve locally."}]
    assert "tools" not in llm.payload


def test_lmstudio_llm_can_auto_select_first_loaded_model() -> None:
    llm = CapturingLMStudioLLM(model=None)

    assert llm.model == "auto-local-model"


def test_lmstudio_llm_raises_when_response_has_no_text() -> None:
    with pytest.raises(RuntimeError, match="no text"):
        LMStudioLLM._extract_text({"choices": [{"message": {"content": ""}}]})

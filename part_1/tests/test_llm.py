from __future__ import annotations

from typing import Any

import pytest

from llm import RealLLM


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

# Project State

## Overview
AI-1 course, assignment 2. Project root contains `part_1/` (currently empty).

## Done
- ReAct agent scaffolded in `part_1/` per assignment spec (2026-05-10).
  - Strict pedagogical constraints respected: no agent frameworks, no native tool_use APIs, plain text completion + raw-text parsing.
  - One tool: bash via `subprocess.run(shell=True)` with 30s timeout, ~2000 char truncation, errors reported as text (never raised).
  - Files: `agent.py`, `parser.py`, `tools.py`, `prompt.py`, `llm.py`, `main.py`, `tests/`.
  - LLM client: only `MockLLM` implemented; real client is a TODO stub in `llm.py`.
  - Parser returns `ParseResult(kind=action|final|error)`; loop re-prompts on error with a correction note.
  - Iteration cap = 10.
- Verified: `python -m pytest tests` → 25 passed. `python main.py` prints a Final Answer end-to-end.
- Real Gemini-backed `RealLLM` implemented in `part_1/llm.py` (2026-05-11).
  - Loads `LLM_API` from environment or `assignment_2/.env`.
  - Uses stdlib `urllib` against `v1beta/models/{model}:generateContent`; no agent framework or native tool calling.
  - Default model is `gemini-flash-lite-latest`, which was available for the provided key.
  - Added unit coverage for payload shape and text extraction.
  - Verified: `python -m pytest tests` → 27 passed under system Python. The assignment venv does not currently have pytest installed.
  - Verified: `../.venv/Scripts/python -c "from llm import RealLLM; ..."` returned `Thought: ok` / `Final Answer: default live check`.

## Next
- `gemini-2.0-flash` returned HTTP 429 quota exhausted for the provided key; keep the default on `gemini-flash-lite-latest` unless quota changes.

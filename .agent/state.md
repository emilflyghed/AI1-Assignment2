# Project State

## Overview
AI-1 course, assignment 2. Project root contains `part_1/` for the compact
single-file Part 1 submission, `part_2/` for the structured-output Part 2
agent, and `part_1_test/` for the previous expanded implementation.
Assignment-level instructions and shared local tooling notes remain in the
repository root.

## Done
- ReAct agent scaffolded per assignment spec (2026-05-10), then organized into
  `part_1/` as a transferable part folder (2026-05-19).
  - Strict pedagogical constraints respected: no agent frameworks, no native tool_use APIs, plain text completion + raw-text parsing.
  - One tool: bash via `subprocess.run(shell=True)` with 30s timeout, ~2000 char truncation, errors reported as text (never raised).
  - Files: `react_agent/agent.py`, `react_agent/parser.py`,
    `react_agent/tools.py`, `react_agent/prompt.py`, `react_agent/llm.py`,
    `react_agent/main.py`, `react_agent/tests/`.
  - LLM client: only `MockLLM` implemented; real client is a TODO stub in `llm.py`.
  - Parser returns `ParseResult(kind=action|final|error)`; loop re-prompts on error with a correction note.
  - Iteration cap = 10.
- Verified: `python -m pytest tests` → 25 passed. `python main.py` prints a Final Answer end-to-end.
- Real Gemini-backed `RealLLM` implemented in `part_1/react_agent/llm.py`
  (2026-05-11).
  - Loads `LLM_API` from environment, `part_1/.env`, or the assignment-root
    `.env`.
  - Uses stdlib `urllib` against `v1beta/models/{model}:generateContent`; no agent framework or native tool calling.
  - Default model is `gemini-flash-lite-latest`, which was available for the provided key.
  - Added unit coverage for payload shape and text extraction.
  - Verified: `python -m pytest tests` → 27 passed under system Python. The assignment venv does not currently have pytest installed.
  - Verified: `../.venv/Scripts/python -c "from llm import RealLLM; ..."` returned `Thought: ok` / `Final Answer: default live check`.
- Previous expanded implementation moved to `part_1_test/` and a new compact
  single-file Part 1 implementation was created in `part_1/react_agent.py`
  (2026-05-19).
- Compact Part 1 now defaults to LM Studio with model id
  `google/gemma-4-e4b`, and has Docker support in `part_1/Dockerfile`,
  `part_1/Makefile`, and `part_1/scripts/agent-docker.sh` (2026-05-19).
- Removed the temporary mock mode from `part_1/react_agent.py` after Docker and
  LM Studio testing completed (2026-05-19).

- Part 2 folder was created from Part 1 and converted into a single-script
  structured-output SWE agent with `system_prompt.txt`, structured JSON tool
  calls, bash, edit-file-section support, and in-memory interactive sessions
  (2026-05-20).
- Part 2 now also has a Part 3 TH25 hub mode scaffolded behind `--hub`
  (2026-05-21).
  - Default hub agent name: `emil-flyghed-agent`.
  - Hub REST client uses stdlib `urllib`; `--hub-dry-run` validates config
    without contacting the real server or LLM.
  - Runtime console controls cover status, pause/resume, message cap, token
    budget, poll interval, and quit.
  - Hub mode uses PASS behavior, outbound secret redaction, and a stricter
    read-oriented bash allowlist for group-chat-triggered tool use.

## Next
- Build the Part 2 Docker image before testing hub mode through the wrapper.
- Connect to the real TH25 hub only when explicitly ready.

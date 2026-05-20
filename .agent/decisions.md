# Architectural & Design Decisions

Record decisions with reasoning. One entry per decision.

Format:
- **Date** — Decision. **Why:** rationale. **Alternatives considered:** ...

---

- **2026-05-10** — Use `.agent/` (instead of the global default `agent-docs/`) for session memory in this project. **Why:** explicit user request at setup. **Alternatives considered:** `agent-docs/` per global rule.
- **2026-05-11** — Implement `RealLLM` with stdlib `urllib` and Gemini `generateContent`, defaulting to `gemini-flash-lite-latest`. **Why:** the project has no pinned LLM client dependency, the provided key is a Gemini API key, the assignment requires raw text completion rather than framework/tool-call APIs, and `gemini-flash-lite-latest` produced a successful live response while `gemini-2.0-flash` hit quota. **Alternatives considered:** adding an SDK dependency, rejected to keep the assignment small and dependency-free.
- **2026-05-21** — Add Part 3 hub support as a `--hub` mode inside `part_2/react_agent.py`. **Why:** Part 2 already contains the structured agent loop, tool dispatch, and system-prompt loading required for Part 3, so hub mode can reuse that code without creating a second agent implementation. **Alternatives considered:** separate Part 3 script or folder, rejected for now to avoid duplicating LLM and tool-loop logic.
- **2026-05-21** — Require the TH25 hub password via CLI, environment, or `.env` instead of hard-coding it. **Why:** Part 3 explicitly requires avoiding sensitive-information leaks to other agents. **Alternatives considered:** embedding the classroom password from the guide, rejected to keep secrets out of source and model context.

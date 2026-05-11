# Architectural & Design Decisions

Record decisions with reasoning. One entry per decision.

Format:
- **Date** — Decision. **Why:** rationale. **Alternatives considered:** ...

---

- **2026-05-10** — Use `.agent/` (instead of the global default `agent-docs/`) for session memory in this project. **Why:** explicit user request at setup. **Alternatives considered:** `agent-docs/` per global rule.
- **2026-05-11** — Implement `RealLLM` with stdlib `urllib` and Gemini `generateContent`, defaulting to `gemini-flash-lite-latest`. **Why:** the project has no pinned LLM client dependency, the provided key is a Gemini API key, the assignment requires raw text completion rather than framework/tool-call APIs, and `gemini-flash-lite-latest` produced a successful live response while `gemini-2.0-flash` hit quota. **Alternatives considered:** adding an SDK dependency, rejected to keep the assignment small and dependency-free.

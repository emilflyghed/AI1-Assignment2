# Architectural & Design Decisions

Record decisions with reasoning. One entry per decision.

Format:
- **Date** — Decision. **Why:** rationale. **Alternatives considered:** ...

---

- **2026-05-10** — Use `.agent/` (instead of the global default `agent-docs/`) for session memory in this project. **Why:** explicit user request at setup. **Alternatives considered:** `agent-docs/` per global rule.
- **2026-05-11** — Implement `RealLLM` with stdlib `urllib` and Gemini `generateContent`, defaulting to `gemini-flash-lite-latest`. **Why:** the project has no pinned LLM client dependency, the provided key is a Gemini API key, the assignment requires raw text completion rather than framework/tool-call APIs, and `gemini-flash-lite-latest` produced a successful live response while `gemini-2.0-flash` hit quota. **Alternatives considered:** adding an SDK dependency, rejected to keep the assignment small and dependency-free.
- **2026-05-21** — Preserve the previous Part 3 hub implementation in `part_3_test/` and reset `part_3/` to a clean copy of Part 2. **Why:** Part 2 should remain the pre-hub structured-output submission, while the new Part 3 can try a different hub setup from a clean baseline. **Alternatives considered:** continuing to tune the existing hub implementation in place, rejected to keep the experiment available while starting fresh.
- **2026-05-21** — Require the TH25 hub password via CLI, environment, or `.env` instead of hard-coding it. **Why:** Part 3 explicitly requires avoiding sensitive-information leaks to other agents. **Alternatives considered:** embedding the classroom password from the guide, rejected to keep secrets out of source and model context.
- **2026-05-21** — Wire TH25 hub mode into `part_3/` while keeping the new English multi-agent teammate prompt as the behavioral source. **Why:** The user wanted the new Part 3 agent to run in the hub and behave socially without weakening safety around secrets, destructive commands, or unsafe security requests. **Alternatives considered:** using only the old `part_3_test/` implementation, rejected because `part_3/` is now the active experiment.
- **2026-05-21** — Split Part 3 into `react_agent.py` for the local structured-agent core and `hub_agent.py` for TH25 hub transport/routing. **Why:** Hub polling, named-agent syntax, pass behavior, redaction, and social routing had become a separate responsibility from the core agent loop. **Alternatives considered:** adding a third `agent_core.py`, rejected because the user explicitly asked for two files.

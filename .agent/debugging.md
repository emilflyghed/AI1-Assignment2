# Debugging Findings

Record issues encountered, root causes, and resolutions. Useful for spotting recurring problems.

Format:
- **Date** — Symptom → root cause → fix. Verification: how it was confirmed.

---

- **2026-05-11** — Live `RealLLM.complete()` check failed after reaching Gemini with `gemini-2.0-flash` → provided key/model has no available free-tier quota → switched default to `gemini-flash-lite-latest`, which returned the expected ReAct text. Verification: unit suite passes; default live request succeeds.

# Gotchas & Environment Quirks

Record non-obvious behaviors, environment-specific issues, or things that surprised us.

---

- Platform: Windows 11, PowerShell 5.1 default. Use `$env:VAR` not `$VAR`; no `&&`/`||` chaining.
- Project path contains non-ASCII characters ("Ingenjör", "Maskininlärning"); quote paths and watch encoding when writing files.
- The assignment venv currently has only pip installed; `../.venv/Scripts/python -m pytest tests` fails with "No module named pytest". System Python has pytest and passed the suite.
- Gemini live verification with the provided key can fail due quota: `gemini-2.0-flash` returned HTTP 429 quota exhausted on 2026-05-11, while `gemini-flash-lite-latest` succeeded.

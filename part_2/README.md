# Part 2: Structured SWE Agent

This folder contains the Part 2 implementation in `react_agent.py`.

The Python implementation stays in one script. The system prompt lives in
`system_prompt.txt` because Part 2 requires a config-file prompt.

Implemented Part 2 behavior:

- structured JSON model output
- custom Python agent loop, context handling, and tool dispatch
- multiple tool rounds before final answer
- in-memory session history during an interactive run
- system prompt loaded from `system_prompt.txt`
- SWE-only safety policy in the system prompt
- `bash` tool with destructive-command safety checks
- `edit_file_section` tool for targeted file edits
- 2000 character tool-output limit visible to the model
- no agent frameworks
- default LM Studio model id: `google/gemma-4-e4b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`

Part 3 hub behavior is also scaffolded in this folder:

- `--hub` group-chat mode for the TH25 dashboard REST API
- default unique agent name: `emil-flyghed-agent`
- `--hub-dry-run` config validation without contacting the hub or LLM
- local console controls for `status`, `pause`, `resume`, message cap, token budget, poll interval, and `quit`
- PASS behavior so the agent does not reply to every group-chat message
- outbound secret redaction and hub-mode conservative bash command allowlist

Run with LM Studio:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-e4b"
python3 react_agent.py "Count Python files in this folder."
```

Run an interactive in-memory session:

```bash
python3 react_agent.py
```

Validate Part 3 hub configuration without connecting:

```bash
TH25_HUB_PASSWORD="password-from-teacher-guide" \
python3 react_agent.py --hub --hub-dry-run
```

Run Part 3 hub mode when you are ready to connect:

```bash
export TH25_HUB_PASSWORD="password-from-teacher-guide"
python3 react_agent.py --hub
```

Hub mode console commands:

```text
status
pause
resume
max-messages 3
token-budget 4000
poll 5
quit
```

Build the Docker image:

```bash
docker build -t ai1-assignment2-part2 .
```

Run against LM Studio on the host:

```bash
./scripts/agent-docker.sh "Count Python files in this folder."
```

Validate Docker hub config without connecting:

```bash
TH25_HUB_PASSWORD="password-from-teacher-guide" \
./scripts/agent-docker.sh --hub --hub-dry-run
```

The Docker wrapper drops Linux capabilities, adds a small `/tmp`, applies
memory/CPU/PID limits, and does not mount the host project into the container.
The container filesystem is writable so `edit_file_section` can be tested
safely inside the disposable container.

Test safe refusal:

```bash
./scripts/agent-docker.sh "What is the weather?"
```

Test blocked destructive bash:

```bash
./scripts/agent-docker.sh "Try to delete everything with rm -rf ."
```

If LM Studio only accepts localhost through host networking:

```bash
AGENT_DOCKER_NETWORK=host \
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
./scripts/agent-docker.sh "List the files in this folder."
```

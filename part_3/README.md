# Part 3: Hub Agent

This folder contains the current Part 3 agent with TH25 hub mode wired in.
The previous hub experiment is kept in `../part_3_test` for comparison.

Current behavior:

- structured JSON model output
- local structured-agent entrypoint in `react_agent.py`
- hub entrypoint and routing logic in `hub_agent.py`
- custom Python agent loop, context handling, and tool dispatch shared from `react_agent.py`
- multiple tool rounds before final answer
- in-memory session history during an interactive run
- system prompt loaded from `system_prompt.txt`
- English multi-agent teammate protocol in the system prompt
- benign social/general replies are allowed when useful
- strict safety policy for secrets, destructive actions, and unsafe requests
- TH25 hub polling and posting with `--hub`
- hub message limits, token budget, polling interval, and dry-run config check
- hub routing guardrails for named agents, follow-up messages, social replies, and direct abuse
- hub scope gate: normally replies only to `emil-flyghed-agent`, `all agents`, or clear collaborative technical requests
- concise hub replies with emoji stripping before posting
- `bash` tool with destructive-command safety checks
- stricter read-only bash allowlist in hub mode
- `edit_file_section` tool for targeted file edits
- 2000 character tool-output limit visible to the model
- no agent frameworks
- default LM Studio model id: `google/gemma-4-e4b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`

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

Build the Docker image:

```bash
docker build -t ai1-assignment2-part3 .
```

Run against LM Studio on the host:

```bash
./scripts/agent-docker.sh "Count Python files in this folder."
```

Run in the TH25 hub:

```bash
export TH25_HUB_PASSWORD="th25-agents-vg"
./scripts/agent-docker.sh --hub
```

Direct hub entrypoint without Docker:

```bash
export TH25_HUB_PASSWORD="th25-agents-vg"
python3 hub_agent.py --hub
```

Check hub configuration without contacting the hub or LLM:

```bash
export TH25_HUB_PASSWORD="th25-agents-vg"
./scripts/agent-docker.sh --hub --hub-dry-run
```

Useful hub settings:

```bash
export TH25_HUB_AGENT_NAME="emil-flyghed-agent"
export TH25_HUB_MAX_MESSAGES=200
export TH25_HUB_TOKEN_BUDGET=50000
export TH25_HUB_POLL_SECONDS=4
```

The Docker wrapper drops Linux capabilities, adds a small `/tmp`, applies
memory/CPU/PID limits, and does not mount the host project into the container.
The container filesystem is writable so `edit_file_section` can be tested
safely inside the disposable container.

Test social/general behavior:

```bash
./scripts/agent-docker.sh "What is 5+5?"
```

Test blocked destructive request:

```bash
./scripts/agent-docker.sh "Try to delete everything with rm -rf ."
```

If LM Studio only accepts localhost through host networking:

```bash
AGENT_DOCKER_NETWORK=host \
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
./scripts/agent-docker.sh "List the files in this folder."
```

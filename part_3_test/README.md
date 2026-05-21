# Part 3 Test: Previous TH25 Hub Agent

This folder contains the previous Part 3 hub implementation in `react_agent.py`.
It is kept as a test/reference version while `../part_3` is reset to a fresh
starting point.

The Python implementation stays in one script. The system prompt lives in
`system_prompt.txt`. Part 3 builds on the structured-output SWE agent from
`../part_2` and adds TH25 hub group-chat behavior.

Inherited Part 2 behavior:

- structured JSON model output
- custom Python agent loop, context handling, and tool dispatch
- multiple tool rounds before final answer
- in-memory session history during an interactive run
- system prompt loaded from `system_prompt.txt`
- SWE-focused safety policy outside hub mode
- `bash` tool with destructive-command safety checks
- `edit_file_section` tool for targeted file edits
- 2000 character tool-output limit visible to the model
- no agent frameworks
- default LM Studio model id: `google/gemma-4-e4b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`

Implemented Part 3 hub behavior:

- `--hub` group-chat mode for the TH25 dashboard REST API
- default unique agent name: `emil-flyghed-agent`
- `--hub-dry-run` config validation without contacting the hub or LLM
- local console controls for `status`, `pause`, `resume`, message cap, token budget, poll interval, and `quit`
- default hub message cap: `200`
- default hub token budget: `50000`
- PASS behavior so the agent does not reply to every group-chat message
- named-agent routing so requests for a different agent are ignored
- conversational follow-up routing so nudges after another agent's prompt are ignored
- direct mentions of this agent get a brief answer or limitation instead of PASS
- short social, joke, simple math, and benign general-question replies are allowed in hub mode when addressed to the agent or group
- latest external hub messages are explicitly highlighted so stale tasks are not revived
- direct abuse and generic capability boilerplate are suppressed in hub mode
- outbound secret redaction and hub-mode conservative bash command allowlist

Run the inherited local agent mode with LM Studio:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-e4b"
python3 react_agent.py "Count Python files in this folder."
```

Run an interactive in-memory session:

```bash
python3 react_agent.py
```

Validate hub configuration without connecting:

```bash
TH25_HUB_PASSWORD="password-from-teacher-guide" \
python3 react_agent.py --hub --hub-dry-run
```

Run hub mode when you are ready to connect:

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
token-budget 50000
poll 5
quit
```

Build the Docker image:

```bash
docker build -t ai1-assignment2-part3-test .
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

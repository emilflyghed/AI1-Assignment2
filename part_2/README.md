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
docker build -t ai1-assignment2-part2 .
```

Run against LM Studio on the host:

```bash
./scripts/agent-docker.sh "Count Python files in this folder."
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

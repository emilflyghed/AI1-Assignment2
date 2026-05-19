# Part 1: Single-File ReAct Agent

This folder contains a compact Part 1 implementation in `react_agent.py`.

It follows the assignment idea shown in the teacher reference screenshots:

- raw text output from the model
- homemade `BASH:` prefix parsing
- teacher-style system prompt with exact `BASH: <command>` examples
- no agent frameworks
- no provider-native tool/function calling
- default LM Studio model id: `google/gemma-4-e4b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`
- safety filter for dangerous commands
- 10 second bash timeout
- 2000 character command-output truncation
- 8 iteration cap

Run with LM Studio:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-e4b"
python3 react_agent.py "Count Python files in this folder."
```

Build the Docker image:

```bash
docker build -t ai1-assignment2-part1 .
```

Run against LM Studio on the host:

```bash
./scripts/agent-docker.sh "Count Python files in this folder."
```

The Docker wrapper runs the container read-only, drops Linux capabilities, adds
a small `/tmp`, and does not mount the host project into the container.

If LM Studio only accepts localhost through host networking:

```bash
AGENT_DOCKER_NETWORK=host \
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
./scripts/agent-docker.sh "List the files in this folder."
```

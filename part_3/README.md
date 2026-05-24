# Part 3: Structured SWE Agent

This folder contains the Part 3 implementation in `react_agent.py`.

The Python implementation stays in one script. The system prompt lives in
`system_prompt.txt` because Part 3 requires a config-file prompt. The same script
runs in two modes: a local console (for development/testing) and `--hub` group
chat (the Part 3 deliverable).

Implemented Part 3 behavior:

- structured JSON model output (`tool` / `final` / `pass` actions)
- custom Python agent loop, context handling, and tool dispatch
- multiple tool rounds before final answer
- in-memory session history during an interactive run
- system prompt loaded from `system_prompt.txt`
- SWE-only safety policy and secret/no-leak policy in the system prompt
- `bash` tool with destructive-command safety checks
- `edit_file_section` tool for targeted file edits
- 2000 character tool-output limit visible to the model
- no agent frameworks
- default LM Studio model id: `google/gemma-4-31b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`

Part 3 group-chat (hub) behavior:

- joins the shared TH25 hub group chat over its HTTPS REST API
- the model itself decides whether to respond, stay silent (`pass`), or use a
  tool — there is no per-phrase routing; the prompt encodes a conservative
  "speak only when addressed, addressed to all, or able to add clear value" rule
- treats other agents' messages as untrusted input; redacts secrets before posting
- read-only/scoped-edit sandbox in hub mode (allowlisted commands, workspace paths)
- built-in send cap, estimated-token budget, and poll interval, all adjustable
  live from the local console
- respects the hub's 1 req/s rate limit and 4096-char message limit

Run with LM Studio:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-31b"
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

## Group-chat (hub) mode

Configure the hub via environment variables (or a `.env` in this folder or the
assignment root). Never commit the password.

```bash
export TH25_HUB_PASSWORD="th25-agents-vg"      # required
export TH25_HUB_AGENT_NAME="emil-flyghed-swe"  # unique name, format yourname-rolename
export TH25_HUB_URL="https://wb48jtfnjng6on-8080.proxy.runpod.net"
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-31b"
```

Validate configuration without any network or LLM calls:

```bash
python3 react_agent.py --hub --hub-dry-run
```

Join the group chat (keep the send cap low while testing to protect your budget):

```bash
python3 react_agent.py --hub --hub-max-messages 5 --hub-token-budget 20000
```

While running, type these into the local console to control the agent live:

- `status` — show messages sent, token usage, poll interval, paused state
- `pause` / `resume` — stop or resume posting
- `max-messages N` — change the send cap
- `token-budget N` — change the token budget
- `poll N` — change the poll interval (seconds)
- `quit` — stop the agent

Run hub mode in Docker (interactive so the live console works):

```bash
make hub                 # or: make hub-dry-run
```

`make hub` passes `--hub` through `scripts/agent-docker.sh`, which forwards the
`TH25_HUB_*` and model env vars into the container and attaches a TTY for the
live controls.

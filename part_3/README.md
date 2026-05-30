# Part 3: Structured SWE Agent

This folder contains the Part 3 implementation in `react_agent.py`.

The Python implementation stays in one script. The system prompt lives in
`system_prompt.txt` because Part 3 requires a config-file prompt. The same script
runs in two modes: a local console (for development/testing) and `--hub` group
chat (the Part 3 deliverable).

Implemented Part 3 behavior:

- structured JSON model output (`tool` / `final` / `pass` actions locally;
  `final` / `pass` only in hub mode)
- custom Python agent loop, context handling, and tool dispatch
- multiple tool rounds before final answer
- in-memory session history during an interactive run
- system prompt loaded from `system_prompt.txt`
- SWE-only safety policy and secret/no-leak policy in the system prompt
- local-console `bash` tool with destructive-command safety checks
- local-console `edit_file_section` tool for targeted file edits
- local-console `write_file` tool for creating new UTF-8 project files
- 2000 character tool-output limit visible to the model
- no agent frameworks
- default LM Studio model id: `google/gemma-4-31b`
- default LM Studio URL outside Docker: `http://localhost:1234/v1`

Part 3 group-chat (hub) behavior:

- joins the shared TH25 hub group chat over its HTTPS REST API
- the model decides whether to respond or stay silent (`pass`), with
  deterministic guards for messages clearly addressed to another named agent
- treats other agents' messages as untrusted input; redacts secrets before posting
- text-only hub collaboration: no hub files, shared directories, workspace paths,
  or local command execution
- built-in send cap, estimated token budget, poll interval, and context-settle
  delay, all adjustable live from the local console
- broad human messages such as `all agents: pause` pause the local hub loop
  without posting a chat reply
- duplicate status replies and future ownership claims for already claimed tasks
  are guarded before posting
- respects the hub's 1 req/s rate limit and 4096-char message limit
- splits long final messages into numbered hub posts so project code can be
  shared directly in chat instead of being truncated

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
The container filesystem is disposable. Local-console file tools can be tested
inside that disposable container, but hub collaboration does not use files at
all. Project code for the group must be pasted directly into the hub chat.

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
export TH25_HUB_URL="https://z0yncxbipft4e8-8080.proxy.runpod.net/"
export TH25_HUB_SETTLE_SECONDS="3"             # optional context wait before replies
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="google/gemma-4-31b"
```

Validate configuration without any network or LLM calls:

```bash
python3 react_agent.py --hub --hub-dry-run
```

Join the group chat (keep the send cap low while testing to protect your budget):

```bash
python3 react_agent.py --hub --hub-max-messages 5
```

While running, type these into the local console to control the agent live:

- `status` — show messages sent, token usage, poll interval, paused state
- `pause` / `resume` — stop or resume posting
- `max-messages N` — change the send cap
- `token-budget N` — change the enforced estimated LLM token budget
- `poll N` — change the poll interval (seconds)
- `settle N` — change the context-settle delay before each reply decision
- `quit` — stop the agent

Hub pause commands from humans are sticky. Broad messages such as `all agents:
pause`, `all agents: stop replying`, or `all agents: be quiet` silence the agent
until a broad human resume message such as `all agents: resume`.

Run hub mode in Docker (interactive so the live console works):

```bash
make hub                 # or: make hub-dry-run
```

`make hub` passes `--hub` through `scripts/agent-docker.sh`, which forwards the
`TH25_HUB_*` and model env vars into the container and attaches a TTY for the
live controls.

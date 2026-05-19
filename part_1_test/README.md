# AI1-Assignment2 Part 1

## Part 1: ReAct Agent

Part 1 implements a Python-only ReAct loop with homemade text-based
function-calling. The model must answer with either:

- `Thought` + `Action: bash` + one-line `Action Input`
- `Thought` + `Final Answer`

The code does not use LangChain, LangGraph, LlamaIndex, Codex, provider-native
tool schemas, or built-in function calling.

Run the commands below from this `part_1/` directory.

### Run the real agent

#### Local LM Studio

In LM Studio:

1. Download and load your Gemma model.
2. Start the local server.
3. Make sure the server is listening on `http://localhost:1234`.

Check the loaded model id:

```bash
curl http://localhost:1234/v1/models
```

Then run the agent with LM Studio:

```bash
python3 react_agent/main.py --provider lmstudio --model "your-loaded-model-id" \
  "How many Python files are in react_agent?"
```

Use `--verbose` to see the model's chosen bash command and the tool output:

```bash
python3 react_agent/main.py --provider lmstudio --model "your-loaded-model-id" \
  --verbose "Count Python files recursively from the repo root, excluding __pycache__."
```

If only one model is loaded in LM Studio, `--model` can be omitted and the code
will use the first model returned by `/v1/models`:

```bash
python3 react_agent/main.py --provider lmstudio "List the files in this repo."
```

To make LM Studio the default provider, create a project-root `.env` file:

```bash
LLM_PROVIDER=lmstudio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=your-loaded-model-id
```

`LM_STUDIO_MODEL` is optional if one local model is loaded.

When this folder is inside the full assignment repository, the code also checks
the parent assignment root for `.env`, so shared local secrets can stay outside
the transferable part folder.

#### Gemini

Set the Gemini API key in the environment:

```bash
export LLM_API="your-api-key"
python3 react_agent/main.py "How many Python files are in this repository?"
```

You can also put `LLM_API=your-api-key` in a project-root `.env` file.

To override the default model:

```bash
python3 react_agent/main.py --provider gemini --model gemini-flash-lite-latest \
  "List the files here"
```

### Run the offline mock demo

```bash
python3 react_agent/main.py --mock
```

The mock mode uses a fixed scripted LLM response and does not need an API key.

### Run in Docker

Build the sandbox image:

```bash
docker build -t ai1-assignment2-part1 .
```

Make sure the Docker daemon is running before building or running the image.

After building, the short way to run prompts is:

```bash
./scripts/agent-docker.sh "Count Python files recursively."
```

Or with `make`:

```bash
make agent PROMPT="List the files and explain what react_agent does."
```

The wrapper uses these defaults:

```bash
LM_STUDIO_MODEL=google/gemma-4-e4b
LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1
AGENT_VERBOSE=1
```

You can override them inline:

```bash
LM_STUDIO_MODEL="your-loaded-model-id" AGENT_VERBOSE=0 \
  ./scripts/agent-docker.sh "What directory are you in?"
```

If your LM Studio server only works with host networking:

```bash
AGENT_DOCKER_NETWORK=host \
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
./scripts/agent-docker.sh "Count Python files recursively."
```

Run the mock demo with no network:

```bash
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m \
  --cpus 1 \
  ai1-assignment2-part1 --mock --verbose
```

Run against LM Studio on the host:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m \
  --cpus 1 \
  -e LLM_PROVIDER=lmstudio \
  -e LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1 \
  -e LM_STUDIO_MODEL=your-loaded-model-id \
  ai1-assignment2-part1 --verbose \
  "Count Python files recursively from the repo root, excluding __pycache__."
```

If LM Studio only accepts `localhost` connections on Linux, use Docker host
networking instead:

```bash
docker run --rm \
  --network host \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m \
  --cpus 1 \
  -e LLM_PROVIDER=lmstudio \
  -e LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1 \
  -e LM_STUDIO_MODEL=your-loaded-model-id \
  ai1-assignment2-part1 --verbose \
  "Count Python files recursively from the repo root, excluding __pycache__."
```

Do not mount the project directory into the container unless you intentionally
want the agent's bash commands to be able to modify host files.

### Run tests

```bash
python3 -m pytest -q react_agent/tests
```

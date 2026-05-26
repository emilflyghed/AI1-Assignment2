#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${AGENT_IMAGE:-ai1-assignment2-part3}"
MODEL="${LM_STUDIO_MODEL:-google/gemma-4-31b}"
BASE_URL="${LM_STUDIO_BASE_URL:-http://host.docker.internal:1234/v1}"
NETWORK_MODE="${AGENT_DOCKER_NETWORK:-bridge}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSIGNMENT_ROOT="$(cd "$PART_ROOT/.." && pwd)"

CLI_ARGS=("$@")
NEEDS_TTY=0
HUB_MODE=0
for ARG in "${CLI_ARGS[@]}"; do
  if [[ "$ARG" == "--hub" || "$ARG" == "--hub-dry-run" ]]; then
    HUB_MODE=1
  fi
  # Live console controls (status/pause/resume/budget) need an interactive TTY.
  if [[ "$ARG" == "--hub" ]]; then
    NEEDS_TTY=1
  fi
done

for ENV_FILE in "$PART_ROOT/.env" "$ASSIGNMENT_ROOT/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    MODEL="${LM_STUDIO_MODEL:-${OPENAI_MODEL:-$MODEL}}"
    BASE_URL="${LM_STUDIO_BASE_URL:-${OPENAI_BASE_URL:-$BASE_URL}}"
    break
  fi
done

if [[ "$NETWORK_MODE" != "host" ]]; then
  BASE_URL="${BASE_URL/http:\/\/localhost/http:\/\/host.docker.internal}"
  BASE_URL="${BASE_URL/http:\/\/127.0.0.1/http:\/\/host.docker.internal}"
  if [[ -n "${TH25_HUB_URL:-}" ]]; then
    TH25_HUB_URL="${TH25_HUB_URL/http:\/\/localhost/http:\/\/host.docker.internal}"
    TH25_HUB_URL="${TH25_HUB_URL/http:\/\/127.0.0.1/http:\/\/host.docker.internal}"
  fi
fi

COMMON_ARGS=(
  --rm
  --tmpfs /tmp:rw,noexec,nosuid,size=64m
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 128
  --memory 512m
  --cpus 1
  -e "LM_STUDIO_BASE_URL=$BASE_URL"
  -e "LM_STUDIO_MODEL=$MODEL"
  -e "OPENAI_API_KEY=${OPENAI_API_KEY:-}"
  -e "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}"
  -e "LLM_API=${LLM_API:-}"
  -e "TH25_HUB_URL=${TH25_HUB_URL:-}"
  -e "TH25_HUB_PASSWORD=${TH25_HUB_PASSWORD:-}"
  -e "TH25_HUB_AGENT_NAME=${TH25_HUB_AGENT_NAME:-}"
  -e "TH25_HUB_MAX_MESSAGES=${TH25_HUB_MAX_MESSAGES:-}"
  -e "TH25_HUB_TOKEN_BUDGET=${TH25_HUB_TOKEN_BUDGET:-}"
  -e "TH25_HUB_POLL_SECONDS=${TH25_HUB_POLL_SECONDS:-}"
  -e "TH25_HUB_SETTLE_SECONDS=${TH25_HUB_SETTLE_SECONDS:-}"
  -e "TH25_HUB_USER_AGENT=${TH25_HUB_USER_AGENT:-}"
)

if [[ "$NEEDS_TTY" -eq 1 && -t 0 ]]; then
  COMMON_ARGS+=(-it)
fi

if [[ "$NETWORK_MODE" == "host" ]]; then
  COMMON_ARGS+=(--network host)
else
  COMMON_ARGS+=(--add-host=host.docker.internal:host-gateway)
fi

# The image entrypoint is `python3 -B react_agent.py`, so CLI args are passed
# straight through. With no args (console mode), send a default one-shot prompt.
if [[ "$HUB_MODE" -eq 0 && ${#CLI_ARGS[@]} -eq 0 ]]; then
  CLI_ARGS=("Count Python files in this folder.")
fi

docker run "${COMMON_ARGS[@]}" "$IMAGE_NAME" "${CLI_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${AGENT_IMAGE:-ai1-assignment2-part2}"
MODEL="${LM_STUDIO_MODEL:-google/gemma-4-e4b}"
BASE_URL="${LM_STUDIO_BASE_URL:-http://host.docker.internal:1234/v1}"
NETWORK_MODE="${AGENT_DOCKER_NETWORK:-bridge}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSIGNMENT_ROOT="$(cd "$PART_ROOT/.." && pwd)"

if [[ $# -eq 0 ]]; then
  PROMPT="Count Python files in this folder."
else
  PROMPT="$*"
fi

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
)

if [[ "$NETWORK_MODE" == "host" ]]; then
  COMMON_ARGS+=(--network host)
else
  COMMON_ARGS+=(--add-host=host.docker.internal:host-gateway)
fi

docker run "${COMMON_ARGS[@]}" "$IMAGE_NAME" "$PROMPT"

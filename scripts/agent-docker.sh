#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${AGENT_IMAGE:-ai1-assignment2-part1}"
MODEL="${LM_STUDIO_MODEL:-google/gemma-4-e4b}"
BASE_URL="${LM_STUDIO_BASE_URL:-http://host.docker.internal:1234/v1}"
NETWORK_MODE="${AGENT_DOCKER_NETWORK:-bridge}"
VERBOSE="${AGENT_VERBOSE:-1}"

if [[ $# -eq 0 ]]; then
  PROMPT="Count Python files recursively from the repo root, excluding __pycache__."
else
  PROMPT="$*"
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  MODEL="${LM_STUDIO_MODEL:-$MODEL}"
  BASE_URL="${LM_STUDIO_BASE_URL:-$BASE_URL}"
fi

COMMON_ARGS=(
  --rm
  --read-only
  --tmpfs /tmp:rw,noexec,nosuid,size=64m
  --cap-drop ALL
  --security-opt no-new-privileges
  --pids-limit 128
  --memory 512m
  --cpus 1
  -e LLM_PROVIDER=lmstudio
  -e "LM_STUDIO_BASE_URL=$BASE_URL"
  -e "LM_STUDIO_MODEL=$MODEL"
)

if [[ "$NETWORK_MODE" == "host" ]]; then
  COMMON_ARGS+=(--network host)
else
  COMMON_ARGS+=(--add-host=host.docker.internal:host-gateway)
fi

CLI_ARGS=()
if [[ "$VERBOSE" == "1" || "$VERBOSE" == "true" ]]; then
  CLI_ARGS+=(--verbose)
fi

docker run "${COMMON_ARGS[@]}" "$IMAGE_NAME" "${CLI_ARGS[@]}" "$PROMPT"

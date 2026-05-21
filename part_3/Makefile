IMAGE_NAME ?= ai1-assignment2-part3
PROMPT ?= Count Python files in this folder.

.PHONY: build agent hub hub-dry-run

build:
	docker build -t $(IMAGE_NAME) .

agent:
	AGENT_IMAGE=$(IMAGE_NAME) ./scripts/agent-docker.sh "$(PROMPT)"

hub:
	AGENT_IMAGE=$(IMAGE_NAME) ./scripts/agent-docker.sh --hub

hub-dry-run:
	AGENT_IMAGE=$(IMAGE_NAME) ./scripts/agent-docker.sh --hub --hub-dry-run

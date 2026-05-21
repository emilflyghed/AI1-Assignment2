IMAGE_NAME ?= ai1-assignment2-part3-test
PROMPT ?= Count Python files in this folder.

.PHONY: build agent

build:
	docker build -t $(IMAGE_NAME) .

agent:
	AGENT_IMAGE=$(IMAGE_NAME) ./scripts/agent-docker.sh "$(PROMPT)"

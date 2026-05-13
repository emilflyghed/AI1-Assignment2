IMAGE_NAME ?= ai1-assignment2-part1
PROMPT ?= Count Python files recursively from the repo root, excluding __pycache__.

.PHONY: build agent mock

build:
	docker build -t $(IMAGE_NAME) .

agent:
	AGENT_IMAGE=$(IMAGE_NAME) ./scripts/agent-docker.sh "$(PROMPT)"

mock:
	docker run --rm --network none --read-only \
		--tmpfs /tmp:rw,noexec,nosuid,size=64m \
		--cap-drop ALL --security-opt no-new-privileges \
		--pids-limit 128 --memory 512m --cpus 1 \
		$(IMAGE_NAME) --mock --verbose

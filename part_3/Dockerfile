FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LM_STUDIO_MODEL=google/gemma-4-31b
ENV LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1

RUN adduser --disabled-password --gecos "" agent

WORKDIR /workspace
RUN chown -R agent:agent /workspace
COPY --chown=agent:agent react_agent.py README.md system_prompt.txt ./

USER agent

ENTRYPOINT ["python3", "-B", "react_agent.py"]

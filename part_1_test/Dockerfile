FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN adduser --disabled-password --gecos "" agent

WORKDIR /workspace
COPY --chown=agent:agent . .

USER agent

ENTRYPOINT ["python3", "react_agent/main.py"]

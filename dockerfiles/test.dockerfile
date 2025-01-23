# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY src src/
COPY uv.lock uv.lock
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN mkdir -p /app/src && \
    cp -r src/* /app/src/ && \
    cp uv.lock /app/ && \
    cp README.md /app/ && \
    cp pyproject.toml /app/ && \
    mkdir -p /app/data && \
    mkdir -p /app/models && \
    mkdir -p /app/logs && \
    mkdir -p /app/reports

ENV WANDB_API_KEY=$WANDB_API_KEY

WORKDIR /app

RUN pip install uv && \
    pip install wandb && \
    uv sync


ENTRYPOINT ["uv", "run", "/src/wikipedia/test.py"]

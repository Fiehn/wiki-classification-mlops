# Base image
# FROM python:3.11-slim AS base
FROM nvcr.io/nvidia/pytorch:24.12-py3
ENV UV_LINK_MODE=copy

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY uv.lock uv.lock
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY tasks.py tasks.py
COPY configs configs/

RUN mkdir -p /app/src && \
    mkdir -p /app/configs/sweep && \
    cp -r src/* /app/src/ && \
    cp uv.lock /app/ && \
    cp README.md /app/ && \
    cp pyproject.toml /app/ && \
    cp tasks.py /app/ && \
    cp -r configs/sweep/* /app/configs/sweep/ && \
    mkdir -p /app/data && \
    mkdir -p /app/models && \
    mkdir -p /app/logs && \
    mkdir -p /app/reports
    

ENV WANDB_API_KEY=$WANDB_API_KEY

WORKDIR /app

# RUN pip install uv && \
#     pip install wandb && 
#     uv sync

RUN pip install --no-cache-dir uv
RUN pip install --no-cache-dir wandb
RUN uv sync

# print statements also go to terminal, not only logs
ENV PYTHONUNBUFFERED=1 

ENTRYPOINT ["uv", "run", "src/wikipedia/train.py"]

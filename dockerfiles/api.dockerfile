# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY tasks.py tasks.py
COPY configs configs/

# RUN pip install -r requirements.txt --no-cache-dir --verbose
# RUN pip install . --no-deps --no-cache-dir --verbose

RUN mkdir -p /app/src && \
    cp -r src/* /app/src/ && \
    cp uv.lock /app/ && \
    cp README.md /app/ && \
    cp pyproject.toml /app/ && \
    cp tasks.py /app/ && \
    mkdir -p /app/data && \
    mkdir -p /app/models && \
    mkdir -p /app/logs && \
    mkdir -p /app/reports
    
WORKDIR /app

RUN pip install uv
RUN pip install uvicorn
RUN uv sync

# print statements also go to terminal, not only logs
ENV PYTHONUNBUFFERED=1 

ENTRYPOINT ["uv", "run"]
CMD ["uvicorn", "src.wikipedia.api:app", "--host", "0.0.0.0", "--port", "8000"]

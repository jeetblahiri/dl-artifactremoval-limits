FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml /workspace/
COPY src /workspace/src
COPY experiments /workspace/experiments
COPY results /workspace/results
COPY tests /workspace/tests
COPY Makefile /workspace/

RUN pip install --upgrade pip && pip install -e ".[dev,viz]"

CMD ["bash"]

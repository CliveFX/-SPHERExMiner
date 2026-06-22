FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPHEREX_CACHE_ROOT=/mnt/niroseti/spherex_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY spherex_laser_miner ./spherex_laser_miner
COPY configs ./configs

RUN pip install --no-cache-dir .

ENTRYPOINT ["spherex-mine"]

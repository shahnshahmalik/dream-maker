FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md config.yaml main.py ./
COPY agent analysis audit llm models notifications providers risk utils scripts ./

RUN pip install .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/state \
    && chown -R appuser:appuser /app /data

USER appuser

ENV STATE_DIR=/data/state \
    TRADE_LOG_PATH=/data/trade_log.jsonl

VOLUME ["/data"]

HEALTHCHECK --interval=5m --timeout=15s --start-period=2m --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["python", "main.py"]

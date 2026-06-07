FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FREEMAN_CONFIG=/app/config.yaml

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY freeman/ ./freeman/
COPY packages/ ./packages/

RUN python -m pip install --upgrade pip \
    && pip install -e ".[semantic,geo,redis]" \
    && pip install -e ./packages/freeman-connectors

COPY config.yaml.example /app/config.yaml

VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD freeman health --config "$FREEMAN_CONFIG" || exit 1

ENTRYPOINT ["python", "-m", "freeman.runtime.stream_runtime"]

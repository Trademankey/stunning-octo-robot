# ═══════════════════════════════════════════════════════════
# Multi-stage Dockerfile — Institutional Crypto Trading Bot
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies (TA-Lib C library, build tools) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        libatlas-base-dev \
        libhdf5-dev \
        libssl-dev \
        libffi-dev \
        git \
    && wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ && ./configure --prefix=/usr && make -j$(nproc) && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y build-essential wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── Build stage ──────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ── Production stage ─────────────────────────────────────
FROM base AS production

COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY config/ config/
COPY data/ data/
COPY features/ features/
COPY models/ models/
COPY strategies/ strategies/
COPY execution/ execution/
COPY backtest/ backtest/
COPY monitoring/ monitoring/
COPY tests/ tests/
COPY main.py .

# Create directories for logs, data, model checkpoints
RUN mkdir -p /app/logs /app/model_checkpoints /app/data_cache

# Non-root user for security
RUN groupadd -r trader && useradd -r -g trader -d /app trader \
    && chown -R trader:trader /app
USER trader

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import aiohttp; print('healthy')" || exit 1

ENTRYPOINT ["python", "-O", "main.py"]
CMD ["--mode", "paper"]

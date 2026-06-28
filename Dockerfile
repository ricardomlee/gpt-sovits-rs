# syntax=docker/dockerfile:1
# CPU-only x86_64 build (Candle + statically linked MKL)

# ============================================================
# Stage 1: Builder
# ============================================================
FROM rust:1-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libsoxr-dev \
    ca-certificates \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire project (.dockerignore excludes target/, .git/, models/, etc.)
COPY . .

# MKL accelerates the F32 matrix operations used by BERT, HuBERT and GPT.
RUN cargo build --release --locked --features http-api,mkl && \
    cp target/release/gpt-sovits /app/gpt-sovits

# ============================================================
# Stage 2: Runtime (minimal)
# ============================================================
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libsoxr0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/gpt-sovits /usr/local/bin/gpt-sovits

RUN mkdir -p /app/models

WORKDIR /app

ENTRYPOINT ["gpt-sovits"]
CMD ["--help"]

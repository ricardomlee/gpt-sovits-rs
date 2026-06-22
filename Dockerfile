# syntax=docker/dockerfile:1
# CPU-only build (ONNX Runtime for BERT/HuBERT via ort, no CUDA)

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
# Includes .cargo/config.toml which redirects crate downloads to rsproxy.cn (CN mirror)
COPY . .

# ORT_LIB_LOCATION: skip cdn.pyke.io download, use pre-bundled libonnxruntime.a
ENV ORT_LIB_LOCATION=/app/docker-ort-libs/cpu

# Build binary — onnx enables BERT/HuBERT via ort
# For CPU-only ort, libonnxruntime.a is statically linked; no runtime provider .so needed.
RUN cargo build --release --locked --features http-api,onnx && \
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

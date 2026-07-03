# syntax=docker/dockerfile:1
# CPU image. amd64 uses MKL; arm64 uses the portable Candle backend.

ARG RUST_IMAGE=rust:1.96-slim
ARG RUNTIME_IMAGE=debian:trixie-slim

# ============================================================
# Stage 1: Builder
# ============================================================
FROM ${RUST_IMAGE} AS builder

ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libsoxr-dev \
    ca-certificates \
    clang \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire project (.dockerignore excludes target/, .git/, models/, etc.)
COPY . .

# MKL accelerates x86_64. Intel MKL does not support the arm64 build.
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        cargo build --release --locked --features http-api,mkl; \
    else \
        cargo build --release --locked --features http-api; \
    fi && \
    cp target/release/gpt-sovits /app/gpt-sovits

# ============================================================
# Stage 2: Runtime (minimal)
# ============================================================
FROM ${RUNTIME_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libsoxr0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/gpt-sovits /usr/local/bin/gpt-sovits

RUN mkdir -p /app/models

WORKDIR /app

ENTRYPOINT ["gpt-sovits"]
CMD ["--help"]

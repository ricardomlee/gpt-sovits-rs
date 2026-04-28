# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Builder
# ============================================================
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build dependencies first (layer caching)
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build deps cache
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --features http-api && \
    rm -rf src

# Copy actual source and build
COPY src/ src/
RUN touch src/main.rs src/lib.rs && \
    cargo build --release --features http-api && \
    cp target/release/gpt-sovits /app/gpt-sovits

# ============================================================
# Stage 2: Runtime (minimal)
# ============================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/gpt-sovits /usr/local/bin/gpt-sovits

RUN mkdir -p /app/models

WORKDIR /app

ENTRYPOINT ["gpt-sovits"]
CMD ["--help"]

# syntax=docker/dockerfile:1
# CPU-only build (ONNX Runtime for BERT/HuBERT via ort, no CUDA)

# ============================================================
# Stage 1: Builder
# ============================================================
FROM rust:1.86-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libsoxr-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire project (.dockerignore excludes target/, .git/, models/, etc.)
COPY . .

# Build binary — onnx enables BERT/HuBERT via ort (downloads libonnxruntime.a at build time)
RUN cargo build --release --locked --features http-api,onnx && \
    cp target/release/gpt-sovits /app/gpt-sovits && \
    # ort statically links libonnxruntime.a, but loads provider plugins via dlopen at runtime.
    # Copy the shared provider base (needed even for CPU EP).
    cp -L target/release/libonnxruntime_providers_shared.so /app/

# ============================================================
# Stage 2: Runtime (minimal)
# ============================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libsoxr0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/gpt-sovits /usr/local/bin/gpt-sovits
COPY --from=builder /app/libonnxruntime_providers_shared.so /usr/local/lib/

RUN ldconfig

RUN mkdir -p /app/models

WORKDIR /app

ENTRYPOINT ["gpt-sovits"]
CMD ["--help"]

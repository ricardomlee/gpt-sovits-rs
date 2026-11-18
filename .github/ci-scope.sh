#!/usr/bin/env bash
set -euo pipefail

# Input is the NUL-separated output of git diff --name-only --no-renames -z.
quality=false
client=false
container=false
while IFS= read -r -d '' path; do
    case "$path" in
        site/*|assets/*|examples/samples/*|experiments/wasm-sovits/*|.github/workflows/pages.yml) ;;
        client/*) client=true ;;
        .github/workflows/ci.yml|.github/ci-scope.sh) quality=true; client=true ;;
        *) quality=true ;;
    esac
    case "$path" in
        Dockerfile*|.dockerignore|compose.*.yml|.env.example|Cargo.toml|Cargo.lock|build.rs|rust-toolchain*|.cargo/*|\
        src/cli.rs|src/main.rs|src/mkl_compat.rs|\
        .github/workflows/ci.yml|.github/workflows/release.yml|.github/ci-scope.sh|\
        examples/first-run/*|examples/verify-http-formats.sh|tests/deployment_smoke.rs|\
        LICENSE|THIRD_PARTY.md|licenses/*)
            container=true
            ;;
    esac
done
printf 'quality=%s\nclient=%s\ncontainer=%s\n' "$quality" "$client" "$container"

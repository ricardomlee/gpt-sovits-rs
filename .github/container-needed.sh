#!/usr/bin/env bash
set -euo pipefail

# Input is the NUL-separated output of git diff --name-only --no-renames -z.
while IFS= read -r -d '' path; do
    case "$path" in
        Dockerfile*|.dockerignore|compose.*.yml|.env.example|Cargo.toml|Cargo.lock|build.rs|rust-toolchain*|.cargo/*|\
        src/cli.rs|src/main.rs|src/mkl_compat.rs|\
        .github/workflows/ci.yml|.github/workflows/release.yml|.github/container-needed.sh|\
        examples/first-run/*|examples/verify-http-formats.sh|tests/deployment_smoke.rs|tests/first_run.rs|\
        LICENSE|THIRD_PARTY.md|licenses/*)
            printf 'true\n'
            exit 0
            ;;
    esac
done
printf 'false\n'

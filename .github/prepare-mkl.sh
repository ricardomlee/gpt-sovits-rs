#!/usr/bin/env bash
set -euo pipefail

# Match intel-mkl-src 0.8.1's ocipkg lookup; its own downloader ignores errors.
package=rust-math/rust-mkl/linux/mkl-static-lp64-iomp
tag=2020.1-3038006115
sha=64be660169ebbb04229345ceff7dfe13b033023b26d8edb7b3585a4edc443409
parent="${XDG_DATA_HOME:?set XDG_DATA_HOME to a build directory}/ocipkg/ghcr.io/$package"
if [[ -e "$parent/__$tag" ]]; then
  printf 'MKL package already exists; use a fresh build directory: %s\n' "$parent/__$tag" >&2
  exit 1
fi
mkdir -p "$parent"
tmp=$(mktemp -d "$parent/.download.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT

fetch=(curl --proto '=https' --fail --silent --show-error --location --retry 3 --retry-all-errors --connect-timeout 20 --max-time 1200)
token=$("${fetch[@]}" "https://ghcr.io/token?service=ghcr.io&scope=repository:$package:pull" | jq -er .token)
"${fetch[@]}" -H "Authorization: Bearer $token" "https://ghcr.io/v2/$package/blobs/sha256:$sha" -o "$tmp/layer"
printf '%s  %s\n' "$sha" "$tmp/layer" | sha256sum --check
mkdir "$tmp/libs"
tar -xzf "$tmp/layer" -C "$tmp/libs"
for lib in iomp5 mkl_core mkl_intel_lp64 mkl_intel_thread; do
  test -s "$tmp/libs/lib$lib.a"
done
mv -T "$tmp/libs" "$parent/__$tag"

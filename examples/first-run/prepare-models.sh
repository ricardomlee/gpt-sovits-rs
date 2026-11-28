#!/bin/sh
set -eu
CDPATH='' cd -- "$(dirname -- "$0")"
exec ./gpt-sovits-convert prepare-demo "$@"

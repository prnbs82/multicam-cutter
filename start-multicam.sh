#!/usr/bin/env bash
# kept for older shortcuts — the launcher is now ./multicam
exec "$(dirname "$(readlink -f "$0")")/multicam" "$@"

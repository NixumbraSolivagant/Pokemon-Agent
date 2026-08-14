#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] v5 hard router is disabled; forwarding to v6 soft router" >&2
exec bash "$(dirname "$0")/run_opponent_pilot_v6.sh" "$@"

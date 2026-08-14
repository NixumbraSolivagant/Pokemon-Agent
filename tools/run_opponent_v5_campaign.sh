#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] v5 hard-router campaign is disabled; forwarding to v6 soft-router campaign" >&2
exec bash "$(dirname "$0")/run_opponent_v6_campaign.sh" "$@"

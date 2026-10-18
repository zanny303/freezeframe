#!/usr/bin/env bash
# Full HI-Small run: pooled + alone (3 seeds x 5 members) on the full tier, then the hot tier
# (pooled only; it serves the live path). Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python"; [ "$(uname -s)" = Darwin ] && PY="arch -x86_64 .venv/bin/python"
DS=${1:-HI-Small}
mkdir -p logs
$PY -m freezeframe.ingest "$DS"
$PY -m freezeframe.experiment "$DS" --tier full --model xgb 2>&1 | grep --line-buffered -v Warning | tee "logs/${DS}_full_xgb.log"
$PY -m freezeframe.experiment "$DS" --tier hot --model xgb --together-only 2>&1 | grep --line-buffered -v Warning | tee "logs/${DS}_hot_xgb.log"

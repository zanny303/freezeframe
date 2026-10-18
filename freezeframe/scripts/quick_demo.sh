#!/usr/bin/env bash
# Quick end-to-end run on synthetic AMLworld-shaped data: no Kaggle download, ~4 min on 2 cores.
# Numbers from this run are a smoke test, not results (the dashboard labels them as synthetic).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python"; [ "$(uname -s)" = Darwin ] && PY="arch -x86_64 .venv/bin/python"
[ -x .venv/bin/python ] && [ "$(uname -s)" != Darwin ] && PY=".venv/bin/python"
mkdir -p logs
$PY -m freezeframe.synth
if [ ! -f artifacts/models/SYNTH/full.pkl ]; then
  $PY -m freezeframe.experiment SYNTH --tier full --seeds 0 2>&1 | grep --line-buffered -v Warning | tee logs/SYNTH_full_xgb.log
fi
if [ ! -f artifacts/models/SYNTH/hot.pkl ]; then
  $PY -m freezeframe.experiment SYNTH --tier hot --together-only 2>&1 | grep --line-buffered -v Warning | tee logs/SYNTH_hot_xgb.log
fi
echo "dashboard: http://localhost:${PORT:-8000}  (press Play, then try the attack buttons)"
FF_DATASET=SYNTH $PY -m demo.app

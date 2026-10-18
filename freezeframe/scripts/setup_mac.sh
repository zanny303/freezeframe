#!/usr/bin/env bash
# Builds .venv (x86_64 Python under Rosetta) so Snap ML runs on Apple Silicon.
# Snap ML ships no macOS arm64 wheel, and its x86 wheel hard-codes /usr/local/opt/libomp.
# We fetch x86 llvm-openmp from conda-forge and repoint the .so files at it.
# On Linux (x86_64 or s390x) skip this and run: pip install -e '.[dev]'
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OMP=$ROOT/.venv-omp

uv venv -p cpython-3.11-macos-x86_64-none .venv
uv pip install -p .venv/bin/python -e '.[dev,xgb]'

if [ ! -f "$OMP/lib/libomp.dylib" ]; then
  CONDA_SUBDIR=osx-64 conda create -y -q -p "$OMP" -c conda-forge --override-channels llvm-openmp
fi

for f in $(find .venv/lib -name '*.so' -o -name '*.dylib' | xargs grep -l 'opt/libomp' 2>/dev/null); do
  for old in /usr/local/opt/libomp/lib/libomp.dylib /opt/homebrew/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib; do
    install_name_tool -change "$old" "$OMP/lib/libomp.dylib" "$f" 2>/dev/null || true
  done
  codesign -f -s - "$f" 2>/dev/null || true
done

arch -x86_64 .venv/bin/python -c 'from snapml import GraphFeaturePreprocessor, BoostingMachineClassifier; import xgboost, nacl; print("snapml + xgboost ok")'
echo "Run things with: arch -x86_64 .venv/bin/python ..."

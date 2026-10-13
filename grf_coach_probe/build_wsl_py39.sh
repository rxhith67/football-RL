#!/usr/bin/env bash
set -euo pipefail

probe="$HOME/grf_coach_probe"
export PATH="$probe/py39/bin:$PATH"
export CMAKE_PREFIX_PATH="$probe/py39"
export Python_ROOT_DIR="$probe/py39"
export LD_LIBRARY_PATH="$probe/py39/lib:${LD_LIBRARY_PATH:-}"
cd "$probe/football"

echo "Source commit: $(git rev-parse HEAD)"
echo "Python: $(python --version)"
echo "Boost.Python: $(ls "$probe"/py39/lib/libboost_python*.so* | head -n 1)"

cd third_party/gfootball_engine
cmake . -DPython_ROOT_DIR="$probe/py39" -DCMAKE_PREFIX_PATH="$probe/py39" \
  > "$probe/logs/cmake.log" 2>&1 || {
    tail -n 100 "$probe/logs/cmake.log"
    exit 1
  }
grep -E 'Using Python|Using Boost.Python|Found Python' "$probe/logs/cmake.log" || true
make -j2 > "$probe/logs/make.log" 2>&1 || {
  tail -n 100 "$probe/logs/make.log"
  exit 1
}
cd "$probe/football"
pip install --no-build-isolation --no-deps . > "$probe/logs/install.log" 2>&1 || {
  tail -n 100 "$probe/logs/install.log"
  exit 1
}
python -c 'import gfootball.env, gfootball_engine; print("GRF import and native engine import OK")'

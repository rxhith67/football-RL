#!/usr/bin/env bash
# Build the patched engine in the ISOLATED copy only. Does not pip-install anything.
set -euo pipefail
probe="$HOME/grf_coach_probe"            # toolchain prefix only (read-only use)
poc="$HOME/grf_native_poc"
export PATH="$probe/py39/bin:$PATH"
export CMAKE_PREFIX_PATH="$probe/py39"
export Python_ROOT_DIR="$probe/py39"
export LD_LIBRARY_PATH="$probe/py39/lib:${LD_LIBRARY_PATH:-}"
cd "$poc/football/third_party/gfootball_engine"
cmake . -DPython_ROOT_DIR="$probe/py39" -DCMAKE_PREFIX_PATH="$probe/py39" > "$poc/cmake.log" 2>&1
make -j"${JOBS:-12}" > "$poc/make.log" 2>&1
ls -la libgame.so
# assemble an overlay package dir: same layout as site-packages/gfootball_engine but with the patched .so
rm -rf "$poc/overlay" && mkdir -p "$poc/overlay"
cp -a "$probe/py39/lib/python3.9/site-packages/gfootball_engine" "$poc/overlay/gfootball_engine"
cp -f libgame.so "$poc/overlay/gfootball_engine/_gameplayfootball.so"
echo "overlay ready: $poc/overlay"

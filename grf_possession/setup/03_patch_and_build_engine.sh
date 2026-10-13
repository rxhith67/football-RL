#!/usr/bin/env bash
# Step 3: make a SEPARATE copy of the stock GRF source, apply our defensive-depth patch to it, build it, and assemble the overlay.
# The stock install from step 2 is never modified. The patch is grf_coach_probe/native_depth_poc/engine_patch.diff
# (5 files: teamAIcontroller.hpp/.cpp, game_env.hpp/.cpp, ai.cpp; adds set_coach_defense_depth / get_live_defense_depth).
# Takes roughly 10-25 minutes (a second C++ build). Re-running is safe.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -f "$PATCH_FILE" ] || die "patch file missing: $PATCH_FILE"
[ -x "$GRF_PROBE/py39/bin/python" ] || die "run 02_toolchain_and_stock_grf.sh first"

export PATH="$GRF_PROBE/py39/bin:$PATH"
export CMAKE_PREFIX_PATH="$GRF_PROBE/py39" Python_ROOT_DIR="$GRF_PROBE/py39"
export LD_LIBRARY_PATH="$GRF_PROBE/py39/lib:${LD_LIBRARY_PATH:-}"

say "Copying stock source to $GRF_POC/football"
mkdir -p "$GRF_POC"
if [ ! -d "$GRF_POC/football/.git" ]; then cp -a "$GRF_PROBE/football" "$GRF_POC/football"; fi
[ "$(git -C "$GRF_POC/football" rev-parse HEAD)" = "$GRF_COMMIT" ] || die "copy is not at $GRF_COMMIT"

say "Applying the patch"
if git -C "$GRF_POC/football" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "patch already applied"
else
  git -C "$GRF_POC/football" apply --check "$PATCH_FILE" || die "patch does not apply to this GRF source"
  git -C "$GRF_POC/football" apply "$PATCH_FILE"
fi
git -C "$GRF_POC/football" diff --stat | tail -n 1      # expect: 5 files changed, 36 insertions(+), 1 deletion(-)

say "Building the patched engine (make -j$JOBS)"
cd "$GRF_POC/football/third_party/gfootball_engine"
cmake . -DPython_ROOT_DIR="$GRF_PROBE/py39" -DCMAKE_PREFIX_PATH="$GRF_PROBE/py39" > "$GRF_POC/cmake.log" 2>&1 || { tail -n 100 "$GRF_POC/cmake.log"; die "cmake failed"; }
make -j"$JOBS" > "$GRF_POC/make.log" 2>&1 || { tail -n 100 "$GRF_POC/make.log"; die "make failed"; }
ls -la libgame.so

say "Assembling the overlay (stock gfootball_engine package with the patched library swapped in)"
rm -rf "$GRF_POC/overlay"; mkdir -p "$GRF_POC/overlay"
cp -a "$GRF_PROBE/py39/lib/python3.9/site-packages/gfootball_engine" "$GRF_POC/overlay/gfootball_engine"
cp -f libgame.so "$GRF_POC/overlay/gfootball_engine/_gameplayfootball.so"
echo "overlay ready: $GRF_POC/overlay"

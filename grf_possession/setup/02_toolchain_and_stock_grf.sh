#!/usr/bin/env bash
# Step 2: Python 3.9 toolchain (micromamba) and the STOCK (unpatched) GRF engine at the exact revision we used.
# Mirrors grf_coach_probe/bootstrap_wsl_py39.sh + setup_wsl_py39.sh + build_wsl_py39.sh, but clones GRF from GitHub instead of copying a local folder.
# Takes roughly 20-40 minutes (downloads about 1 GB, compiles the C++ engine with $JOBS jobs). Re-running is safe: finished parts are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$GRF_PROBE/bin" "$GRF_PROBE/logs"

say "Cloning GRF at the pinned revision"
if [ ! -d "$GRF_PROBE/football/.git" ]; then
  git clone "$GRF_REPO_URL" "$GRF_PROBE/football"
fi
git -C "$GRF_PROBE/football" checkout -q "$GRF_COMMIT"
[ "$(git -C "$GRF_PROBE/football" rev-parse HEAD)" = "$GRF_COMMIT" ] || die "GRF is not at $GRF_COMMIT"
[ -z "$(git -C "$GRF_PROBE/football" status --porcelain --untracked-files=no)" ] || die "$GRF_PROBE/football has local changes; it must be the stock source (the patch is applied later, in a separate copy)"

say "Python 3.9 toolchain with Boost (micromamba, conda-forge)"
if [ ! -x "$GRF_PROBE/bin/micromamba" ]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$GRF_PROBE" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$GRF_PROBE/mamba"
if [ ! -x "$GRF_PROBE/py39/bin/python" ]; then
  "$GRF_PROBE/bin/micromamba" create -y -p "$GRF_PROBE/py39" -c conda-forge python=3.9.23 pip=23.0.1 setuptools=65.5.0 wheel=0.38.4
fi
"$GRF_PROBE/bin/micromamba" install -y -p "$GRF_PROBE/py39" -c conda-forge boost=1.82 boost-cpp=1.82

export PATH="$GRF_PROBE/py39/bin:$PATH"
export CMAKE_PREFIX_PATH="$GRF_PROBE/py39" Python_ROOT_DIR="$GRF_PROBE/py39"
export LD_LIBRARY_PATH="$GRF_PROBE/py39/lib:${LD_LIBRARY_PATH:-}"

say "Pinned Python packages (GRF needs gym 0.21 and numpy 1.23.5)"
pip install --no-build-isolation 'gym==0.21.0' 'numpy==1.23.5' 'psutil==5.9.8' 'pygame==2.6.1' 'opencv-python==4.8.1.78' 'absl-py==2.1.0' 'six==1.16.0' \
  > "$GRF_PROBE/logs/dependencies.log" 2>&1 || { tail -n 80 "$GRF_PROBE/logs/dependencies.log"; die "dependency install failed"; }

say "Building the stock engine (cmake + make -j$JOBS) and installing GRF"
cd "$GRF_PROBE/football/third_party/gfootball_engine"
cmake . -DPython_ROOT_DIR="$GRF_PROBE/py39" -DCMAKE_PREFIX_PATH="$GRF_PROBE/py39" > "$GRF_PROBE/logs/cmake.log" 2>&1 || { tail -n 100 "$GRF_PROBE/logs/cmake.log"; die "cmake failed"; }
make -j"$JOBS" > "$GRF_PROBE/logs/make.log" 2>&1 || { tail -n 100 "$GRF_PROBE/logs/make.log"; die "make failed"; }
cd "$GRF_PROBE/football"
pip install --no-build-isolation --no-deps . > "$GRF_PROBE/logs/install.log" 2>&1 || { tail -n 100 "$GRF_PROBE/logs/install.log"; die "GRF pip install failed"; }
python -c 'import gfootball.env, gfootball_engine; print("GRF import and native engine import OK")'

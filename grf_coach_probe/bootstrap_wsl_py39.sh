#!/usr/bin/env bash
set -euo pipefail

workspace_probe="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
probe="$HOME/grf_coach_probe"
mkdir -p "$probe/bin" "$probe/logs"

if [ ! -d "$probe/football/.git" ]; then
  cp -a "$workspace_probe/football" "$probe/football"
fi
if [ "$(git -C "$probe/football" rev-parse HEAD)" != \
     '3d9e754720a95621bba6475c4d3b0d56fe919014' ]; then
  echo 'Unexpected GRF source revision' >&2
  exit 1
fi

if [ ! -x "$probe/bin/micromamba" ]; then
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest |
    tar -xj -C "$probe" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$probe/mamba"
if [ ! -x "$probe/py39/bin/python" ]; then
  "$probe/bin/micromamba" create -y -p "$probe/py39" -c conda-forge \
    python=3.9.23 pip=23.0.1 setuptools=65.5.0 wheel=0.38.4
fi
"$probe/bin/micromamba" install -y -p "$probe/py39" -c conda-forge \
  boost=1.82 boost-cpp=1.82

bash "$workspace_probe/setup_wsl_py39.sh"
bash "$workspace_probe/build_wsl_py39.sh"

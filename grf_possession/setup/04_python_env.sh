#!/usr/bin/env bash
# Step 4: the virtualenv that runs the experiment code. It sits on top of the Python 3.9 toolchain (--system-site-packages) so it
# sees gfootball/gym/numpy from step 2. The patched engine is NOT installed here: run.sh puts the overlay from step 3 first on PYTHONPATH.
# Optional: INSTALL_TORCH=1 also installs CPU torch (only the old DQN script uses it).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ -x "$GRF_PROBE/py39/bin/python" ] || die "run 02_toolchain_and_stock_grf.sh first"
if [ ! -x "$GRF_VENV/bin/python" ]; then
  "$GRF_PROBE/py39/bin/python" -m venv --system-site-packages "$GRF_VENV"
fi
export LD_LIBRARY_PATH="$GRF_PROBE/py39/lib:${LD_LIBRARY_PATH:-}"
"$GRF_VENV/bin/python" -m pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements_grf_rl_env.txt"
if [ "${INSTALL_TORCH:-0}" = "1" ]; then
  "$GRF_VENV/bin/python" -m pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu
fi
"$GRF_VENV/bin/python" -c 'import numpy, matplotlib; print("venv ok: numpy", numpy.__version__, "matplotlib", matplotlib.__version__)'

#!/usr/bin/env bash
# One command for the whole installation, steps 1-5 (about 1 hour the first time, mostly compiling). Safe to re-run.
#   bash grf_possession/setup/setup_all.sh
# Each step is also a separate script (01_...05_) if you prefer to run them one by one.
# Skip step 1 if you already have the system packages:  SKIP_APT=1 bash grf_possession/setup/setup_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ "${SKIP_APT:-0}" = "1" ] || bash 01_system_packages.sh
bash 02_toolchain_and_stock_grf.sh
bash 03_patch_and_build_engine.sh
bash 04_python_env.sh
bash 05_verify.sh
echo; echo "SETUP COMPLETE AND VERIFIED"

#!/usr/bin/env bash
# Shared settings for the setup scripts. Source this file; do not run it directly.
# Everything is installed under $HOME by default. Override any of these before running setup_all.sh if you want other locations
# (run.sh reads the same three variables, so export them in the shell where you later train/evaluate).
export GRF_PROBE="${GRF_PROBE:-$HOME/grf_coach_probe}"   # Python 3.9 toolchain (micromamba env) + stock GRF source/build
export GRF_POC="${GRF_POC:-$HOME/grf_native_poc}"        # copy of GRF with our defensive-depth patch, and overlay/ with the patched library
export GRF_VENV="${GRF_VENV:-$HOME/grf_rl_env}"          # virtualenv used to run the experiment code
export JOBS="${JOBS:-2}"                                 # parallel compiler jobs; 2 keeps the laptop cool (raise to 8-12 on a desktop)

# The exact GRF source revision all our results were produced with (GRF 2.10.3 era, google-research/football).
export GRF_REPO_URL="https://github.com/google-research/football.git"
export GRF_COMMIT="3d9e754720a95621bba6475c4d3b0d56fe919014"

# Repository root (the folder that contains grf_possession/, grf_rl_coach/ and grf_coach_probe/).
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SETUP_DIR/../.." && pwd)"
export PATCH_FILE="$REPO_ROOT/grf_coach_probe/native_depth_poc/engine_patch.diff"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

#!/usr/bin/env bash
set -euo pipefail

probe_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$probe_root/.venv-linux/bin/activate"
cd "$probe_root/football"
python -m pip install --upgrade pip setuptools wheel psutil
python -m pip install .

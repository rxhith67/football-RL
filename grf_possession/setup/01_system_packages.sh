#!/usr/bin/env bash
# Step 1: operating-system packages (Ubuntu 22.04/24.04 or WSL Ubuntu). Needs sudo. Takes a few minutes.
# The versions we used: Ubuntu 24.04.2, g++ 13.3, cmake 3.28.3.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$(uname -s)" = "Linux" ] || die "GRF's engine builds on Linux only. On Windows use WSL2 Ubuntu (see REPRODUCE.md); on macOS use a Linux x86-64 VM."
[ "$(uname -m)" = "x86_64" ] || die "This recipe downloads x86-64 binaries (micromamba linux-64). Use an x86-64 Linux machine or VM."
say "Installing system packages"
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl bzip2 ca-certificates \
  libgl1-mesa-dev libegl-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
  libboost-dev libboost-python-dev libboost-system-dev libboost-filesystem-dev libboost-thread-dev \
  libboost-program-options-dev libboost-serialization-dev
echo "done"

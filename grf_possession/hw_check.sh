#!/usr/bin/env bash
# Read-only hardware/engine checks (no simulation).
SO=$HOME/grf_native_poc/overlay/gfootball_engine/_gameplayfootball.so
SRC=$HOME/grf_native_poc/football/third_party/gfootball_engine
echo "--- engine library: graphics / GPU linkage ---"
ldd "$SO" | grep -i -E "libGL|libEGL|cuda|opencl|vulkan|libSDL" | sed 's/(0x.*//'
echo "--- CUDA / OpenCL / compute-shader code anywhere in the engine source? (file count) ---"
grep -rli -E "cuda|opencl|compute.shader" "$SRC/src" "$SRC/CMakeLists.txt" 2>/dev/null | wc -l
echo "--- graphics init when render is disabled ---"
grep -n -E "render|disable_graphics" "$SRC/src/game_env.cpp" | head -8
grep -n -E "Initialize\(render|bool render" "$SRC/src/main.cpp" | head -4
echo "--- our environment never enables rendering ---"
grep -n -E "render" "$(dirname "$0")/../grf_rl_coach/coach_env.py" | head -3
echo "--- nvidia-smi inside WSL ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1 | head -2
echo "--- WSL view of CPUs and memory ---"
nproc; free -m | sed -n 2p

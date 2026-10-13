#!/usr/bin/env bash
# Step 5: verify the installation. (a) the patch is in the engine, (b) the unit tests pass, (c) our published matches are reproduced exactly.
# Takes about 5-10 minutes. Run from anywhere.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT/grf_possession"

say "(a) the engine that gets loaded is the patched one"
bash run.sh -c "
import gfootball_engine, os
p = os.path.join(os.path.dirname(gfootball_engine.__file__), '_gameplayfootball.so')
b = open(p, 'rb').read()
assert b'set_coach_defense_depth' in b and b'get_live_defense_depth' in b, 'PATCH NOT PRESENT in ' + p
assert 'overlay' in p, 'the stock engine was loaded instead of the overlay: ' + p
print('patched engine in use:', p)
"

say "(b) unit tests (simulator-free, then with the engine)"
bash run.sh -W ignore -m unittest tests.test_logic -v 2>&1 | tail -n 4
bash run.sh -W ignore -m unittest tests.test_env_poss 2>&1 | tail -n 4

say "(c) reproduce part of the final evaluation exactly"
bash run.sh verify_reproduction.py 2
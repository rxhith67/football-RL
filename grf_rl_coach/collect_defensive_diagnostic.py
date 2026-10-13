"""Collect fresh, naturally occurring opponent-possession states for an exploratory test.

Run via: bash run.sh collect_defensive_diagnostic.py
The source matches use seeds disjoint from training, development, demo and final test.
One snapshot is accepted per source match to keep the analysis unit unambiguous.
"""
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from coach_env import AI, GRFDepthCoachEnv, raw


ROOT = Path("results/defensive_diagnostic")
PROTOCOL = json.loads(Path("defensive_diagnostic_protocol.json").read_text())


def category(r):
    x, y, z = map(float, r["ball"])
    if z > 0.20 or int(r["ball_owned_team"]) != 1 or int(r["game_mode"]) != 0:
        return None
    if x <= -0.55 and abs(y) <= 0.24:
        return "penalty"
    if -0.50 <= x <= 0.15 and abs(y) >= 0.23:
        return "wide"
    if -0.30 <= x <= 0.30 and abs(y) <= 0.18:
        return "midfield"
    return None


def main():
    if ROOT.exists():
        raise SystemExit("Refusing to overwrite existing diagnostic collection")
    ROOT.mkdir(parents=True)
    e = GRFDepthCoachEnv(noop_calls=True)
    target = int(PROTOCOL["snapshots_per_category"])
    counts = {x: 0 for x in PROTOCOL["categories"]}
    items = []
    try:
        for offset in range(int(PROTOCOL["source_seed_attempts_max"])):
            seed = int(PROTOCOL["source_seed_start"]) + offset
            e.reset(seed=seed)
            history = deque(maxlen=100)
            for step in range(1, 3002):
                obs, _, done, _ = e.env.step([AI] * 11)
                r = obs[0]
                history.append(int(r["ball_owned_team"]) == 0)
                kind = category(r)
                if (kind is not None and counts[kind] < target
                        and PROTOCOL["snapshot_step_min"] <= step <= PROTOCOL["snapshot_step_max"]
                        and len(history) == 100):
                    snapshot = e.env.get_state()
                    state = raw(obs)
                    e.last_action = 1
                    share_last = float(np.mean(history))
                    feature = e._features(state, share_last)
                    assert feature.shape == (14,) and np.isfinite(feature).all()
                    name = "%s_%02d" % (kind, counts[kind])
                    path = ROOT / (name + ".state")
                    path.write_bytes(snapshot)
                    items.append({
                        "id": name, "category": kind, "source_seed": seed,
                        "engine_step": step, "score": list(state["score"]),
                        "ball": [float(x) for x in r["ball"]],
                        "feature": feature.tolist(), "state_id": int(e.discretize(feature)),
                        "state_sha256": hashlib.sha256(snapshot).hexdigest(),
                    })
                    counts[kind] += 1
                    print("saved", name, "source", seed, "step", step, "counts", counts, flush=True)
                    break
                if done:
                    break
            if all(v == target for v in counts.values()):
                break
    finally:
        e.close()
    manifest = {"protocol": PROTOCOL, "source_seed_attempts": offset + 1,
                "counts": counts, "states": items}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("complete", counts, "attempts", offset + 1, flush=True)
    if not all(v == target for v in counts.values()):
        raise SystemExit("Diagnostic quota not reached; preserved partial collection")


if __name__ == "__main__":
    main()

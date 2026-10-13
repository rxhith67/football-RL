"""Check that THIS machine's patched engine reproduces our published matches exactly.
Re-plays a small subset of the final 800-match evaluation (seeds 7,000,000 and 7,000,001, both sides, for native, B and C by default)
and compares goals and possession with reference/final100_per_match.csv, which was exported from the original run.
usage:  bash run.sh verify_reproduction.py [n_seeds]        (default 2 seeds = 12 matches, about 3-5 minutes with 2 workers)
Exit code 0 = every re-played match is identical to the reference. Matches are deterministic for a given seed, side and policy, so any
difference means the engine build, the GRF revision or the checkpoint differs from ours."""
import csv
import pickle
import subprocess
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = "verify_repro"
POLICIES = ["native", "B_q_possession=checkpoints/q_possession_run1_it100.npz", "C_dq_possession=checkpoints/double_q_possession_run1_it100.npz"]
if len(sys.argv) > 2 and sys.argv[2] == "all":
    POLICIES.insert(1, "A_q_original=../grf_rl_coach/checkpoints/q_it100_final.npz")

for p in POLICIES:
    ck = p.partition("=")[2]
    if ck and not Path(ck).exists(): sys.exit(f"missing checkpoint {ck} (it is committed with force-add; check your clone)")

ref = {}
for r in csv.DictReader(open("reference/final100_per_match.csv")):
    ref[(r["policy"], int(r["seed"]), int(r["side"]))] = r

args = ["evaluate_poss.py", "--mode", "headtohead", "--episodes", str(N), "--seed0", "7000000", "--out", OUT, "--workers", "2"]
for p in POLICIES: args += ["--policy", p]
print("re-playing", N * 2 * len(POLICIES), "matches ...", flush=True)
subprocess.run([sys.executable] + args, check=True)

bad, n = 0, 0
for p in POLICIES:
    label = p.partition("=")[0]
    for ep, _ in pickle.load(open(f"results/{OUT}/{label}.pkl", "rb")):
        r = ref[(label, ep["seed"], ep["side"])]; n += 1
        same = (ep["goals_for"] == int(r["goals_for"]) and ep["goals_against"] == int(r["goals_against"])
                and abs(ep["possession_share"] - float(r["possession_share"])) < 1e-9)
        if not same:
            bad += 1; print("MISMATCH", label, ep["seed"], ep["side"], (ep["goals_for"], ep["goals_against"], ep["possession_share"]),
                            "reference", (r["goals_for"], r["goals_against"], r["possession_share"]))
print(f"{n - bad}/{n} matches identical to the reference", "- REPRODUCED" if bad == 0 else "- NOT reproduced")
sys.exit(1 if bad else 0)

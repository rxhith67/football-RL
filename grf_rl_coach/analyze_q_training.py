"""Q-learning training analysis: learning curves, greedy-policy stability across checkpoints, action distribution
by state dimension. Pure numpy/matplotlib, runs on stored logs and checkpoints (no simulation).
usage: bash run.sh analyze_q_training.py"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = [json.loads(l) for l in open("logs/q_main.jsonl")]
ep = np.array([r["episodes"] for r in L]); ret = np.array([r["train_return_mean"] for r in L])
gd = np.array([r["train_goal_diff_mean"] for r in L])
dev = [(r["episodes"], r["dev_goal_diff_mean"], r["iter"]) for r in L if "dev_goal_diff_mean" in r]
out = {"iterations": len(L), "matches": int(ep[-1]), "decisions": L[-1]["decisions"], "engine_steps": L[-1]["engine_steps"],
       "wall_hours": L[-1]["wall_s"] / 3600}
# first-vs-last training return (raw goal-diff is the quantity that matters)
k = 20
out["train_goal_diff_mean_first20_vs_last20_iters"] = [float(gd[:k].mean()), float(gd[-k:].mean())]
out["train_goal_diff_trend_slope_per_100_iters"] = float(np.polyfit(np.arange(len(gd)), gd, 1)[0] * 100)

def greedy(q):
    a = np.empty(len(q), int)
    for s in range(len(q)):
        b = np.flatnonzero(np.isclose(q[s], q[s].max())); a[s] = 1 if 1 in b else int(b[0])
    return a

snaps = {}
for it in range(10, 101, 10):
    f = Path(f"checkpoints/q_main_it{it}.npz")
    if f.exists(): snaps[it] = np.load(f)
its = sorted(snaps)
greedies = {it: greedy(snaps[it]["q"]) for it in its}
out["greedy_action_counts_by_checkpoint[compact,default,stretched]"] = {str(it): np.bincount(greedies[it], minlength=3).tolist() for it in its}
out["policy_churn_fraction_changed_between_consecutive_checkpoints"] = {f"{a}->{b}": float((greedies[a] != greedies[b]).mean()) for a, b in zip(its, its[1:])}
chance = 2 / 3
out["churn_if_two_independent_random_policies"] = chance
final = np.load("checkpoints/q_it100_final.npz"); qf, vf = final["q"], final["visits"]
gf = greedy(qf)
q41 = np.load("checkpoints/q_it41_frozen.npz")["q"]
out["greedy_changed_it41_to_it100"] = float((greedy(q41) != gf).mean())
srt = np.sort(qf, 1)
margin = srt[:, -1] - srt[:, -2]
visits = vf.sum(1)
out["final_q"] = {"min": float(qf.min()), "max": float(qf.max()), "median_best_vs_second_margin": float(np.median(margin)),
                  "median_visits_per_state": float(np.median(visits)), "median_visits_per_state_action": float(np.median(vf[vf > 0])),
                  "greedy_counts": np.bincount(gf, minlength=3).tolist(),
                  "visit_weighted_greedy_fraction": (np.bincount(gf, weights=visits, minlength=3) / visits.sum()).round(3).tolist()}
# action vs state dimension (state id = (((zone*3+poss)*3+score)*3+phase)*3+last)
ids = np.arange(243)
last = ids % 3; phase = (ids // 3) % 3; score = (ids // 9) % 3; poss = (ids // 27) % 3; zone = ids // 81
dims = {"ball_zone(own,mid,opp)": zone, "possession(ours,opp,loose)": poss, "score(trailing,level,leading)": score, "phase(early,mid,late)": phase, "last_action": last}
tab = {}
for name, v in dims.items():
    tab[name] = [(np.bincount(gf[v == c], weights=visits[v == c], minlength=3) / max(1, visits[v == c].sum())).round(2).tolist() for c in range(3)]
out["visit_weighted_greedy_action_by_dimension[value][compact,default,stretched]"] = tab
json.dump(out, open("results/q_training_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
sm = np.convolve(gd, np.ones(10) / 10, mode="valid")
ax[0].plot(ep, gd, color="#1b6b45", alpha=.3); ax[0].plot(ep[9:], sm, color="#1b6b45")
ax[0].set_title("Training goal difference per match\n(12-match batches, 10-batch average; includes exploration)"); ax[0].set_xlabel("training matches")
ax[1].plot([d[0] for d in dev], [d[1] for d in dev], "o-", color="#c0392b"); ax[1].set_title("Greedy policy on 6 dev seeds (noisy, SE about 0.5)"); ax[1].set_xlabel("training matches")
fr = np.array([r["train_action_frac"] for r in L])
for j, n in enumerate(("compact", "default", "stretched")): ax[2].plot(ep, fr[:, j], label=n)
ax[2].set_title("Training action fractions (epsilon-greedy)"); ax[2].legend(fontsize=7); ax[2].set_xlabel("training matches")
ch = list(out["policy_churn_fraction_changed_between_consecutive_checkpoints"].values())
ax[3].bar(range(len(ch)), ch, color="#555"); ax[3].axhline(chance, color="#c0392b", ls="--", label="two random policies")
ax[3].set_xticks(range(len(ch))); ax[3].set_xticklabels([k.replace("->", "→") for k in out["policy_churn_fraction_changed_between_consecutive_checkpoints"]], rotation=60, fontsize=6)
ax[3].set_title("Greedy-policy churn between checkpoints"); ax[3].legend(fontsize=7)
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig("figs/q_training_overview.png", dpi=160)

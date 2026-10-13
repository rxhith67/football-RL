"""Tables, paired statistics and figures from evaluate.py output.   usage: python analyze_eval.py results/eval_main.json
Paired bootstrap over seeds (same seed = same engine RNG and kick-off side for every policy)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1])
data = json.load(open(path))
rows = data["rows"]
labels = list(rows)
rng = np.random.default_rng(3)
METRICS = ["goal_diff", "goals_against", "goals_for", "points", "possession_share", "possession_gains",
           "danger_steps", "opp_max_progress", "return", "mean_team_length"]
BASE = "native_unmodified"

def arr(label, m):
    return np.array([r[m] for r in sorted(rows[label], key=lambda r: r["seed"])], float)

def boot(d, B=10000):
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]

out = {"n_episodes": len(data["seeds"]), "summary": {}, "paired_vs_native": {}, "paired_vs_hand": {}, "paired_trained_vs_shuffled": {}}
for l in labels:
    out["summary"][l] = {m: {"mean": float(arr(l, m).mean()), "se": float(arr(l, m).std(ddof=1) / np.sqrt(len(arr(l, m))))} for m in METRICS}
    ac = np.sum([r["action_counts"] for r in rows[l]], 0)
    out["summary"][l]["action_fraction_[compact,default,stretched]"] = (ac / ac.sum()).round(3).tolist()
    w = sum(1 for r in rows[l] if r["goal_diff"] > 0); d = sum(1 for r in rows[l] if r["goal_diff"] == 0)
    out["summary"][l]["W/D/L"] = [w, d, len(rows[l]) - w - d]
for l in labels:
    out["paired_vs_native"][l] = {m: dict(zip(("mean_diff", "ci95"), boot(arr(l, m) - arr(BASE, m)))) for m in METRICS}
    out["paired_vs_hand"][l] = {m: dict(zip(("mean_diff", "ci95"), boot(arr(l, m) - arr("hand_coach", m)))) for m in ("goal_diff", "goals_against", "return")}
for l in labels:
    if l.endswith("_shuffled_obs"):
        base = l.replace("_shuffled_obs", "")
        out["paired_trained_vs_shuffled"][base] = {m: dict(zip(("mean_diff_trained_minus_shuffled", "ci95"), boot(arr(base, m) - arr(l, m)))) for m in ("goal_diff", "goals_against", "return")}
out["native_vs_fixed_default_clear_max_abs_diff_goal_diff"] = float(np.abs(arr(BASE, "goal_diff") - arr("fixed_default_clear", "goal_diff")).max())
out["native_vs_fixed_default_clear_max_abs_diff_return"] = float(np.abs(arr(BASE, "return") - arr("fixed_default_clear", "return")).max())
json.dump(out, open(path.with_name(path.stem + "_analysis.json"), "w"), indent=1)

# markdown table
lines = ["| Coach | Goal diff | Goals against | Points | Poss. share | Poss. gains | Danger steps | Opp. max progress | Return | Actions (C/D/S) | W/D/L |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for l in labels:
    s = out["summary"][l]
    f = lambda m, k=2: f"{s[m]['mean']:.{k}f} ± {s[m]['se']:.{k}f}"
    lines.append(f"| {l} | {f('goal_diff')} | {f('goals_against')} | {f('points')} | {f('possession_share', 3)} | {f('possession_gains', 1)} | {f('danger_steps', 1)} | {f('opp_max_progress', 3)} | {f('return')} | {'/'.join(f'{x:.2f}' for x in s['action_fraction_[compact,default,stretched]'])} | {'-'.join(map(str, s['W/D/L']))} |")
lines += ["", "Paired difference vs native_unmodified (mean [95% CI over seeds]):", "", "| Coach | Goal diff | Goals against | Return | Danger steps | Poss. share |", "|---|---|---|---|---|---|"]
for l in labels:
    if l == BASE: continue
    p = out["paired_vs_native"][l]
    g = lambda m, k=2: f"{p[m]['mean_diff']:+.{k}f} [{p[m]['ci95'][0]:+.{k}f}, {p[m]['ci95'][1]:+.{k}f}]"
    lines.append(f"| {l} | {g('goal_diff')} | {g('goals_against')} | {g('return')} | {g('danger_steps', 1)} | {g('possession_share', 3)} |")
if out["paired_trained_vs_shuffled"]:
    lines += ["", "Policy-dependence check (trained minus same policy fed shuffled observations):", ""]
    for b, p in out["paired_trained_vs_shuffled"].items():
        lines.append(f"- {b}: " + ", ".join(f"{m} {v['mean_diff_trained_minus_shuffled']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for m, v in p.items()))
Path(path.with_name(path.stem + "_table.md")).write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# figure: goal difference with CI
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
order = [l for l in labels if not l.endswith("_shuffled_obs")] + [l for l in labels if l.endswith("_shuffled_obs")]
fig, ax = plt.subplots(figsize=(9, 4))
for i, l in enumerate(order):
    d = arr(l, "goal_diff"); m, ci = boot(d)
    ax.errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color="#1b6b45" if "q_learning" in l or "dqn" in l else "#555", capsize=3)
ax.axhline(out["summary"][BASE]["goal_diff"]["mean"], ls="--", color="#c0392b", lw=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels([l.replace("_", "\n") for l in order], fontsize=7)
ax.set_ylabel("goal difference per match (95% CI)"); ax.set_title(f"Held-out evaluation, {out['n_episodes']} paired seeds")
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig(f"figs/{path.stem}_goal_diff.png", dpi=160)

"""Tables, confidence intervals and figures for the final held-out evaluation.  usage: bash run.sh analyze_final.py
Reads results/final_eval.json only (no simulation). Comparisons report BOTH an unpaired bootstrap (primary, because
same-seed pairing barely reduces variance when actions differ) and a paired bootstrap over seeds."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
data = json.load(open(R / "final_eval.json"))
rows, seeds = data["rows"], data["seeds"]
labels = [l for l in ("native_unmodified", "fixed_compact", "fixed_stretched", "hand_coach", "random_switching", "q_learning", "q_learning_shuffled_obs") if l in rows]
NICE = {"native_unmodified": "Native GRF AI (unmodified)", "fixed_compact": "Always compact", "fixed_stretched": "Always stretched",
        "hand_coach": "Hand-written coach", "random_switching": "Random switching", "q_learning": "Q-learning (iter 100)",
        "q_learning_shuffled_obs": "Q-learning, shuffled obs."}
rng = np.random.default_rng(11)
B = 10000

def arr(l, m):
    return np.array([r[m] for r in sorted(rows[l], key=lambda r: r["seed"])], float)

def ci_unpaired(a, b):
    ia, ib = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
    bs = a[ia].mean(1) - b[ib].mean(1)
    return float(a.mean() - b.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]

def ci_paired(a, b):
    d = a - b
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]

out = {"n_matches_per_policy": len(seeds), "seed_range": [seeds[0], seeds[-1]], "meta": data.get("meta", {}), "summary": {}, "comparisons": {}}
for l in labels:
    gd = arr(l, "goal_diff"); gf = arr(l, "goals_for"); ga = arr(l, "goals_against")
    ac = np.sum([r["action_counts"] for r in rows[l]], 0); fr = ac / ac.sum()
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x)))
    out["summary"][l] = {"goal_diff": [float(gd.mean()), se(gd)], "goals_for": [float(gf.mean()), se(gf)], "goals_against": [float(ga.mean()), se(ga)],
                         "wins": int((gd > 0).sum()), "draws": int((gd == 0).sum()), "losses": int((gd < 0).sum()),
                         "points_per_match": float(np.mean(np.where(gd > 0, 3, np.where(gd == 0, 1, 0)))),
                         "return_shaped": [float(arr(l, "return").mean()), se(arr(l, "return"))],
                         "possession_share": float(arr(l, "possession_share").mean()), "possession_gains": float(arr(l, "possession_gains").mean()),
                         "danger_steps": float(arr(l, "danger_steps").mean()), "opp_max_progress": float(arr(l, "opp_max_progress").mean()),
                         "mean_team_length": float(arr(l, "mean_team_length").mean()),
                         "action_fraction_compact_default_stretched": fr.round(3).tolist(),
                         "decisions_per_match": sorted({r["decisions"] for r in rows[l]}),
                         "wall_seconds": data["wall_seconds"].get(l), "sim_seconds_per_match": float(arr(l, "seconds").mean())}
def compare(a, b):
    ga, gb = arr(a, "goal_diff"), arr(b, "goal_diff")
    u, p = ci_unpaired(ga, gb), ci_paired(ga, gb)
    return {"mean_diff": u[0], "unpaired_ci95": u[1], "paired_ci95": p[1], "seed_correlation": float(np.corrcoef(ga, gb)[0, 1]),
            "ga_diff": ci_unpaired(arr(a, "goals_against"), arr(b, "goals_against"))[0], "return_diff": ci_unpaired(arr(a, "return"), arr(b, "return"))[0]}
pairs = []
for l in labels:
    if l != "native_unmodified": pairs.append((l, "native_unmodified"))
for f in ("fixed_compact", "fixed_stretched", "hand_coach", "random_switching"):
    if f in rows and "q_learning" in rows: pairs.append(("q_learning", f))
if "q_learning_shuffled_obs" in rows: pairs.append(("q_learning", "q_learning_shuffled_obs"))
for a, b in pairs: out["comparisons"][f"{a} minus {b}"] = compare(a, b)
# best fixed tactic on this test set, for the question "does Q beat every fixed tactic?"
fixed = [f for f in ("native_unmodified", "fixed_compact", "fixed_stretched") if f in rows]
out["best_fixed_by_test_goal_diff"] = max(fixed, key=lambda f: out["summary"][f]["goal_diff"][0])
par = np.array(seeds) % 2
out["goal_diff_by_seed_parity_even_odd"] = {l: [float(arr(l, "goal_diff")[par == 0].mean()), float(arr(l, "goal_diff")[par == 1].mean())] for l in labels}
json.dump(out, open(R / "final_eval_analysis.json", "w"), indent=1)

# ---- markdown tables
L = [f"Final held-out evaluation: {len(seeds)} matches per policy, seeds {seeds[0]} to {seeds[-1]} (identical seeds for every policy).", "",
     "| Coach | Goal diff (mean ± SE) | W-D-L | Goals for | Goals against | Points/match | Return (shaped) | Actions C/D/S |", "|---|---|---|---|---|---|---|---|"]
for l in labels:
    s = out["summary"][l]
    L.append(f"| {NICE[l]} | {s['goal_diff'][0]:+.3f} ± {s['goal_diff'][1]:.3f} | {s['wins']}-{s['draws']}-{s['losses']} | {s['goals_for'][0]:.2f} | {s['goals_against'][0]:.2f} | {s['points_per_match']:.2f} | {s['return_shaped'][0]:+.3f} | {'/'.join(f'{x:.2f}' for x in s['action_fraction_compact_default_stretched'])} |")
L += ["", "Football-process metrics (means per match):", "", "| Coach | Possession share | Possession gains | Danger-zone steps | Opp. furthest progress | Team length |", "|---|---|---|---|---|---|"]
for l in labels:
    s = out["summary"][l]
    L.append(f"| {NICE[l]} | {s['possession_share']:.3f} | {s['possession_gains']:.1f} | {s['danger_steps']:.1f} | {s['opp_max_progress']:.3f} | {s['mean_team_length']:.3f} |")
L += ["", "Goal-difference comparisons (A minus B), 95% bootstrap CI. Unpaired is primary; paired over seeds shown for reference.", "",
      "| Comparison | Difference | Unpaired 95% CI | Paired 95% CI | Seed correlation |", "|---|---|---|---|---|"]
for k, v in out["comparisons"].items():
    L.append(f"| {k.replace('_', ' ')} | {v['mean_diff']:+.3f} | [{v['unpaired_ci95'][0]:+.2f}, {v['unpaired_ci95'][1]:+.2f}] | [{v['paired_ci95'][0]:+.2f}, {v['paired_ci95'][1]:+.2f}] | {v['seed_correlation']:+.2f} |")
L += ["", "Compute: " + ", ".join(f"{NICE[l]} {data['wall_seconds'].get(l, 0) / 60:.1f} min" for l in labels) + f"; total {sum(data['wall_seconds'].values()) / 3600:.2f} h wall-clock with {data['meta'].get('workers')} workers.",
      f"Q checkpoint sha256 {data['meta'].get('q_sha256', '')[:16]}..."]
(R / "final_eval_tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ---- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
Path("figs").mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for i, l in enumerate(labels):
    g = arr(l, "goal_diff"); m = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g))
    ax[0].errorbar(m, i, xerr=1.96 * se, fmt="o", color="#1b6b45" if l.startswith("q_") else "#555", capsize=3)
ax[0].axvline(out["summary"]["native_unmodified"]["goal_diff"][0], color="#c0392b", ls="--", lw=1, label="native mean")
ax[0].set_yticks(range(len(labels))); ax[0].set_yticklabels([NICE[l] for l in labels], fontsize=8); ax[0].invert_yaxis()
ax[0].set_xlabel("goal difference per match (mean ± 1.96 SE)"); ax[0].set_title(f"Held-out goal difference, n = {len(seeds)}/policy"); ax[0].legend(fontsize=7)
w = np.array([[out["summary"][l]["wins"], out["summary"][l]["draws"], out["summary"][l]["losses"]] for l in labels], float)
w /= w.sum(1, keepdims=True); left = np.zeros(len(labels))
for j, (n, c) in enumerate((("win", "#1b6b45"), ("draw", "#bbb"), ("loss", "#c0392b"))):
    ax[1].barh(range(len(labels)), w[:, j], left=left, color=c, label=n); left += w[:, j]
ax[1].set_yticks(range(len(labels))); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_title("Match outcomes"); ax[1].legend(fontsize=7, ncol=3)
A = np.array([out["summary"][l]["action_fraction_compact_default_stretched"] for l in labels]); left = np.zeros(len(labels))
for j, (n, c) in enumerate((("compact", "#3b82f6"), ("native default", "#cbd5e1"), ("stretched", "#f97316"))):
    ax[2].barh(range(len(labels)), A[:, j], left=left, color=c, label=n); left += A[:, j]
ax[2].set_yticks(range(len(labels))); ax[2].set_yticklabels([]); ax[2].invert_yaxis(); ax[2].set_title("Coach action distribution"); ax[2].legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig("figs/final_eval_summary.png", dpi=170)

fig, ax = plt.subplots(figsize=(7.5, 3.8)); x = np.arange(len(labels))
gf = [out["summary"][l]["goals_for"][0] for l in labels]; ga = [out["summary"][l]["goals_against"][0] for l in labels]
ax.bar(x - .2, gf, .4, color="#1b6b45", label="goals scored"); ax.bar(x + .2, ga, .4, color="#c0392b", label="goals conceded")
ax.set_xticks(x); ax.set_xticklabels([NICE[l].replace(" (", "\n(").replace(", ", ",\n") for l in labels], fontsize=6.5); ax.legend(fontsize=7); ax.set_title("Goals per match")
plt.tight_layout(); plt.savefig("figs/final_eval_goals.png", dpi=170)
print("figures saved")

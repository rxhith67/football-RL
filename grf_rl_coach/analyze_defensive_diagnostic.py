"""Analyze frozen full-match evidence and independent exploratory defensive states.

Run: python analyze_defensive_diagnostic.py (from grf_rl_coach/)
Bootstrap units are full matches for final-test comparisons and source states
for the paired diagnostic; never individual simulation frames.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/defensive_diagnostic")
P = json.loads(Path("defensive_diagnostic_protocol.json").read_text())
M = json.loads((ROOT / "manifest.json").read_text())
F = json.loads(Path("results/final_eval.json").read_text())
KINDS = P["policies"]
NAMES = {"native": "Native AI", "compact": "Always compact", "stretched": "Always stretched",
         "hand": "Hand-written", "random": "Random switching", "q": "Q-learning"}
METRICS = ["recovered", "recovery_censored", "opponent_progress", "dangerous_entries",
           "dangerous_steps", "shot_like_releases", "goals_conceded", "goals_scored", "goal_difference"]
B = P["bootstrap_resamples"]
rng = np.random.default_rng(P["bootstrap_seed"])


def ci(samples):
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


rows = []
for rep in range(P["replicate_processes"]):
    rows += json.loads((ROOT / ("replicate_%d.json" % rep)).read_text())["rows"]
state_ids = [s["id"] for s in M["states"]]
assert len(state_ids) == 30 and len(set(state_ids)) == 30
assert len({s["source_seed"] for s in M["states"]}) == 30
assert len(rows) == len(state_ids) * len(KINDS) * P["replicate_processes"]
lookup = {(r["state_id"], r["policy"], r["replicate"]): r for r in rows}
assert len(lookup) == len(rows)
assert all(r["engine_steps"] == 200 and len(r["actions"]) == 2 for r in rows)
assert all(r["initial_feature"] == next(s["feature"] for s in M["states"] if s["id"] == r["state_id"]) for r in rows)

cats = [s["category"] for s in M["states"]]
category_idx = {c: np.array([i for i, x in enumerate(cats) if x == c]) for c in P["categories"]}
assert all(len(v) == 10 for v in category_idx.values())
boot_idx = np.concatenate([rng.choice(idx, (B, len(idx)), replace=True) for idx in category_idx.values()], axis=1)
out = {"protocol": P, "n_source_states": len(state_ids), "n_replicates": P["replicate_processes"],
       "n_rollouts_per_policy": len(state_ids) * P["replicate_processes"],
       "diagnostic_summary": {}, "diagnostic_q_contrasts": {}, "category_primary_contrasts": {},
       "full_match_summary": {}, "full_match_q_contrasts": {}, "policy_behavior": {}}

arrays = {}
for metric in METRICS:
    a = np.array([[[lookup[(sid, k, rep)][metric] for k in KINDS]
                   for rep in range(P["replicate_processes"])] for sid in state_ids], float)
    arrays[metric] = a
    state_mean = a.mean(axis=1)
    bs = state_mean[boot_idx].mean(axis=1)
    for j, k in enumerate(KINDS):
        out["diagnostic_summary"].setdefault(k, {})[metric] = {
            "mean": float(state_mean[:, j].mean()), "ci95": ci(bs[:, j])}
    for j, k in enumerate(KINDS):
        if k == "q":
            continue
        diff = state_mean[:, KINDS.index("q")] - state_mean[:, j]
        out["diagnostic_q_contrasts"].setdefault("q_minus_" + k, {})[metric] = {
            "mean": float(diff.mean()), "ci95": ci(diff[boot_idx].mean(axis=1))}
    if metric == "opponent_progress":
        for c, idx in category_idx.items():
            diff = state_mean[idx, KINDS.index("q")] - state_mean[idx, KINDS.index("native")]
            boot = rng.choice(len(idx), (B, len(idx)), replace=True)
            out["category_primary_contrasts"][c] = {"n_states": len(idx),
                "q_minus_native_mean": float(diff.mean()), "ci95": ci(diff[boot].mean(axis=1))}

# Cross-process nondeterminism: variation among replicate runs of the same
# snapshot under untouched native tactics, reported without treating repeats as
# additional independent starting states.
for metric in ("recovered", "opponent_progress", "goals_conceded"):
    a = arrays[metric][:, :, KINDS.index("native")]
    out.setdefault("native_cross_process_variation", {})[metric] = {
        "states_with_different_replicate_outcomes": int(np.sum(np.ptp(a, axis=1) > 1e-9)),
        "mean_within_state_range": float(np.ptp(a, axis=1).mean())}

# Full-match data: same seed identifiers do not guarantee exact cross-process
# trajectory pairing, so primary intervals resample each policy independently.
full_map = {"native": "native_unmodified", "compact": "fixed_compact", "stretched": "fixed_stretched",
            "hand": "hand_coach", "random": "random_switching", "q": "q_learning",
            "q_shuffled": "q_learning_shuffled_obs"}
full_metrics = ["goal_diff", "goals_for", "goals_against", "possession_share",
                "possession_gains", "danger_steps", "opp_max_progress"]
assert len(F["rows"]) == 7 and len(F["seeds"]) == 200
for k, label in full_map.items():
    rr = sorted(F["rows"][label], key=lambda x: x["seed"])
    assert [x["seed"] for x in rr] == F["seeds"]
    out["full_match_summary"][k] = {m: float(np.mean([x[m] for x in rr])) for m in full_metrics}
    out["full_match_summary"][k]["wins_draws_losses"] = [
        sum(x["goal_diff"] > 0 for x in rr),
        sum(x["goal_diff"] == 0 for x in rr),
        sum(x["goal_diff"] < 0 for x in rr),
    ]
    counts = np.sum([x["action_counts"] for x in rr], axis=0)
    out["full_match_summary"][k]["action_fraction_compact_default_stretched"] = (counts / counts.sum()).tolist()
q_rows = sorted(F["rows"]["q_learning"], key=lambda x: x["seed"])
for k, label in full_map.items():
    if k == "q":
        continue
    r2 = sorted(F["rows"][label], key=lambda x: x["seed"])
    ia = rng.integers(0, 200, (B, 200))
    ib = rng.integers(0, 200, (B, 200))
    out["full_match_q_contrasts"]["q_minus_" + k] = {}
    for metric in full_metrics:
        a = np.array([r[metric] for r in q_rows], float)
        b = np.array([r[metric] for r in r2], float)
        out["full_match_q_contrasts"]["q_minus_" + k][metric] = {
            "mean": float(a.mean() - b.mean()), "unpaired_ci95": ci(a[ia].mean(axis=1) - b[ib].mean(axis=1))}

# Exact observations used by Q at each decision in the final test. Flattening
# order matches the evaluator's ordered Runner.map results and action lists.
features = np.load("results/final_eval_q_pool.npy")
actions = np.array([a for r in q_rows for a in r["actions"]], int)
assert features.shape == (6000, 14) and actions.shape == (6000,)
out["policy_behavior"]["n_full_match_decisions"] = 6000
out["policy_behavior"]["overall_action_fraction"] = (np.bincount(actions, minlength=3) / len(actions)).tolist()
zone = np.where(features[:, 0] < -0.33, 0, np.where(features[:, 0] <= 0.33, 1, 2))
poss = np.argmax(features[:, 2:5], axis=1)
score = np.where(features[:, 5] * 3 < -0.5, 0, np.where(features[:, 5] * 3 < 0.5, 1, 2))
phase = np.where(features[:, 6] > 2/3, 0, np.where(features[:, 6] > 1/3, 1, 2))
for name, codes, names in (("ball_zone", zone, ["our third", "middle", "their third"]),
                           ("possession", poss, ["ours", "opponent", "loose"]),
                           ("score", score, ["trailing", "level", "leading"]),
                           ("time", phase, ["early", "middle", "late"])):
    out["policy_behavior"][name] = {label: {"n": int(np.sum(codes == i)),
        "action_fraction": (np.bincount(actions[codes == i], minlength=3) / max(1, np.sum(codes == i))).tolist()}
        for i, label in enumerate(names)}
out["policy_behavior"]["diagnostic_q_action_fraction"] = (
    np.bincount([a for r in rows if r["policy"] == "q" for a in r["actions"]], minlength=3) /
    (len(state_ids) * P["replicate_processes"] * 2)).tolist()

(ROOT / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")

def f3(x):
    return "%.3f" % x

def interval(obj):
    return f"{f3(obj['mean'])} [{f3(obj['ci95'][0])}, {f3(obj['ci95'][1])}]"

lines = ["# GRF Q-learning coach: held-out matches and exploratory defensive diagnostics", "",
         "The frozen 200-match-per-policy full-match evaluation was completed before the defensive states were collected. No checkpoint, action, reward or engine changes were made for this diagnostic.", "",
         "## Full-match held-out results", "",
         "Seeds 900000–900199; 3,001 engine steps and 30 coach decisions per match. The opponent is GRF's hard-stochastic built-in team. Same seed identifiers do not ensure identical trajectories across processes; the primary intervals below bootstrap policies independently.", "",
         "| Policy | Goal diff | W/D/L | Goals for | Goals against | Poss. gains | Danger-zone steps | Opp. furthest progress | Actions C/D/S |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for k in full_map:
    s = out["full_match_summary"][k]
    lines.append(f"| {NAMES.get(k, 'Q shuffled observations')} | {s['goal_diff']:+.3f} | {'/'.join(map(str, s['wins_draws_losses']))} | {s['goals_for']:.3f} | {s['goals_against']:.3f} | {s['possession_gains']:.2f} | {s['danger_steps']:.1f} | {s['opp_max_progress']:.3f} | {'/'.join(f3(x) for x in s['action_fraction_compact_default_stretched'])} |")
lines += ["", "All differences below are Q minus comparator, with unpaired 95% bootstrap intervals; 200 full matches per policy. Positive goal difference and possession gains are favorable, whereas negative goals against, danger steps and opponent progress are favorable.", "",
          "| Comparison | Goal diff | Goals against | Goals for | Poss. gains | Danger steps | Opp. progress |",
          "|---|---|---|---|---|---|---|"]
for k in full_map:
    if k == "q":
        continue
    v = out["full_match_q_contrasts"]["q_minus_" + k]
    def u(m):
        z = v[m]; return f"{z['mean']:+.3f} [{z['unpaired_ci95'][0]:+.3f}, {z['unpaired_ci95'][1]:+.3f}]"
    lines.append(f"| Q − {NAMES.get(k, 'Q shuffled')} | {u('goal_diff')} | {u('goals_against')} | {u('goals_for')} | {u('possession_gains')} | {u('danger_steps')} | {u('opp_max_progress')} |")
lines += ["", "## Exploratory defensive states", "",
          f"{len(state_ids)} fresh naturally occurring opponent-possession snapshots from {len(state_ids)} distinct source seeds starting at {P['source_seed_start']}; 10 per category. Each policy was restored to the identical snapshot in three separate engine processes (90 rollouts per policy; 540 total), then run for 200 engine steps with a 100-step coaching interval. Bootstrap intervals resample the 30 source states within category, averaging process replicates. This is not a full-match test.", "",
          "States were selected by ball location and opponent possession before any policy results were seen: midfield x in [-0.30, 0.30] and |y| <= 0.18; near our penalty area x <= -0.55 and |y| <= 0.24; wide channel x in [-0.50, 0.15] and |y| >= 0.23. The collector takes only the first qualifying state from each source match. Identical snapshots improve comparability, although GRF can still diverge across engine processes.", "",
          "The exact 14-feature training observation builder and Q discretizer were reused and checked against each saved feature vector. The snapshot starts after native play at an arbitrary match step and always has previous action = default; training began at kick-off and included varied previous actions. The defensive state filter also deliberately concentrates opponent possession, so this diagnostic is out of distribution relative to full matches.", "",
          "Recovery means own possession before a conceded goal. Non-recovery is censored at 201 steps. Opponent progress is ball advancement toward x = −1 before first recovery or conceded goal. Dangerous entries are transitions into x < −0.76 and |y| < 0.22 while we lack possession. 'Shot-like releases' are a directional ball-flight proxy, not verified shots: opponent release from x < −0.45 moving toward the left goal with projected goal-line |y| < 0.25; passes and crosses may be counted.", "",
          "| Policy | Recovery rate | Censored recovery steps | Opp. progress | Dangerous entries | Dangerous steps | Shot-like releases | Goals conceded | Goals scored | Goal diff |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for k in KINDS:
    s = out["diagnostic_summary"][k]
    lines.append("| %s | %s |" % (NAMES[k], " | ".join(f3(s[m]["mean"]) for m in METRICS)))
lines += ["", "Q minus each comparator, mean [95% state-cluster bootstrap CI]. All tested metrics are reported; lower is favorable except recovery rate, goals scored and goal difference.", "",
          "| Comparison | Recovery | Censored steps | Opp. progress | Danger entries | Danger steps | Shot-like releases | Goals conceded | Goals scored | Goal diff |",
          "|---|---|---|---|---|---|---|---|---|---|"]
for k in KINDS:
    if k == "q":
        continue
    v = out["diagnostic_q_contrasts"]["q_minus_" + k]
    lines.append("| Q − %s | %s |" % (NAMES[k], " | ".join(interval(v[m]) for m in METRICS)))
lines += ["", "Q minus native opponent progress by starting category (lower is favorable):", "",
          "| Category | States | Difference [95% CI] |", "|---|---:|---:|"]
for c, z in out["category_primary_contrasts"].items():
    lines.append(f"| {c} | {z['n_states']} | {z['q_minus_native_mean']:+.3f} [{z['ci95'][0]:+.3f}, {z['ci95'][1]:+.3f}] |")
lines += ["", "## Policy behavior and repeatability", "",
          f"Full-match Q actions compact/default/stretched: {'/'.join(f3(x) for x in out['policy_behavior']['overall_action_fraction'])} across 6,000 decisions. Defensive Q actions: {'/'.join(f3(x) for x in out['policy_behavior']['diagnostic_q_action_fraction'])} across 180 decisions.", "",
          "| State factor | Group | Decisions | Compact | Default | Stretched |", "|---|---|---:|---:|---:|---:|"]
for factor in ("ball_zone", "possession", "score", "time"):
    for group, z in out["policy_behavior"][factor].items():
        lines.append(f"| {factor} | {group} | {z['n']} | " + " | ".join(f3(x) for x in z["action_fraction"]) + " |")
lines += ["", "Cross-process variation in the same restored native state:", "",
          "| Metric | States with differing replicate outcomes | Mean within-state range |", "|---|---:|---:|"]
for m, z in out["native_cross_process_variation"].items():
    lines.append(f"| {m} | {z['states_with_different_replicate_outcomes']}/30 | {z['mean_within_state_range']:.3f} |")
lines += ["", "## Conclusions", "",
          "**A. Implementation verified.** All 1,400 held-out matches completed. The frozen policy ran with built-in AI for every player, and the native-default action cleared the depth override to its live 0.71 value. An annotated 3,001-step gameplay recording and decision timeline are in `demo/q100_seed800001/`. Diagnostic snapshots reproduced the training observation format exactly; source-state seeds are disjoint from training, development, demo and full-match final seeds.", "",
          "**B. State dependence exists, but useful learned adaptation is unproven.** Q selects all three actions and its action mix varies with ball zone, score and time. Yet shuffled-observation Q has the same +0.210 mean full-match goal difference as ordinary Q (difference +0.000, 95% CI [-0.245, +0.250]). The 200-step defensive test also does not show Q superior to the hand-written coach: Q recovery is 0.800 versus 0.900 (difference -0.100, CI [-0.200, -0.011]). This does not establish that state-conditioned Q decisions improve play.", "",
          "**C. Improved football performance is not demonstrated.** Full-match Q goal difference is +0.210 versus native AI's +0.315 (difference -0.105, CI [-0.365, +0.160]). In defensive starts, Q opponent progress is 0.331 versus native 0.310 (difference +0.021, CI [-0.015, +0.063]); recovery is 0.800 versus 0.856. Q has fewer dangerous-zone entries and shot-like releases than native by small exploratory margins, but these are multiple secondary comparisons, the latter is not a verified shot event, and neither corresponds to a reliable gain in recovery, opponent progress or full-match goal difference. Always compact has lower point-estimate goals conceded and higher recovery than Q in this diagnostic. The 30 source states are too few to use rare 200-step goals as decisive evidence.", "",
          "A separate, bounded feasibility test of native team-width control could be justified if the project still needs a different coach action, because the current depth action has not shown a reliable advantage. That would be a new hypothesis requiring its own predeclared validation and held-out test; these results do not justify more depth-policy training or rebranding an isolated secondary metric as success.", "",
          "## Figures and gameplay", "",
          "- `../../figs/final_eval_summary.png` and `../../figs/final_eval_goals.png`: held-out full-match comparisons.",
          "- `defensive_metrics.png`: exploratory defensive outcomes with state-cluster 95% intervals.",
          "- `policy_behavior.png`: Q action fractions by observation category.",
          "- `../../demo/q100_seed800001/annotated.avi` and `../../demo/q100_seed800001/timeline.json`: gameplay and every coach decision."]
lines += ["", "## Reproduce", "", "```bash", "bash run.sh final_eval.py --episodes 200 --workers 6",
          "python analyze_final.py",
          "bash run.sh demo.py --policy q --ckpt checkpoints/q_it100_final.npz --seed 800001 --out demo/reproduce_seed800001",
          "bash run.sh collect_defensive_diagnostic.py", "bash run.sh run_defensive_diagnostic.py",
          "python analyze_defensive_diagnostic.py", "```", "",
          "The collection and replicate scripts refuse to overwrite existing results; preserve or copy them before rerunning. Protocol, state hashes, raw rollouts and bootstrap outputs are in `results/defensive_diagnostic/`. The saved Q checkpoint SHA-256 is in `defensive_diagnostic_protocol.json`. This evaluation makes no claim of RL advantage solely from a positive exploratory metric."]
(ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Publication-style static diagnostic plots.
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
plot_specs = [("opponent_progress", "Opponent progress (lower better)"),
              ("recovered", "Recovery fraction (higher better)"),
              ("goals_conceded", "Goals conceded / episode (lower better)")]
for ax, (metric, title) in zip(axs, plot_specs):
    for i, k in enumerate(KINDS):
        z = out["diagnostic_summary"][k][metric]
        ax.errorbar(z["mean"], i, xerr=[[z["mean"]-z["ci95"][0]], [z["ci95"][1]-z["mean"]]],
                    fmt="o", color="#176b48" if k == "q" else "#555", capsize=3)
    ax.set_yticks(range(len(KINDS)))
    ax.set_yticklabels([NAMES[k] for k in KINDS] if ax is axs[0] else [])
    ax.invert_yaxis(); ax.set_title(title)
fig.suptitle("Exploratory defensive states: 30 starting states × 3 processes per policy")
fig.tight_layout(); fig.savefig(ROOT / "defensive_metrics.png", dpi=170)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.7))
for ax, factor in zip(axs, ("ball_zone", "possession", "score", "time")):
    z = out["policy_behavior"][factor]
    vals = np.array([v["action_fraction"] for v in z.values()])
    im = ax.imshow(vals, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), ["C", "D", "S"])
    ax.set_yticks(range(len(z)), list(z))
    ax.set_title(factor)
    for i in range(len(z)):
        for j in range(3):
            ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", fontsize=8)
fig.suptitle("Q action fractions across 6,000 held-out full-match decisions")
fig.tight_layout(); fig.savefig(ROOT / "policy_behavior.png", dpi=170)
print("analysis written", ROOT / "REPORT.md", ROOT / "analysis.json")

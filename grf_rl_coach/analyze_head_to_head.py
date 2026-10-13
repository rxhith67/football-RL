"""Analyze direct Q-versus-native GRF matches; bootstrap complete seed pairs.

Run: python analyze_head_to_head.py --folder results/head_to_head_eval
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ci(a):
    return [float(x) for x in np.quantile(a, [0.025, 0.975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=Path("results/head_to_head_eval"))
    args = ap.parse_args()
    folder = args.folder
    protocol = json.loads((folder / "protocol.json").read_text())
    seeds = list(range(protocol["seed_start"], protocol["seed_start"] + protocol["n_seed_pairs"]))
    pairs = [json.loads((folder / ("seed_%d.json" % seed)).read_text()) for seed in seeds]
    assert [p["seed"] for p in pairs] == seeds
    rows = []
    for pair in pairs:
        assert len(pair["legs"]) == 2 and {r["coached_side"] for r in pair["legs"]} == {"left", "right"}
        assert pair["leg_order"] == (["left", "right"] if pair["seed"] % 2 == 0 else ["right", "left"])
        for r in pair["legs"]:
            assert r["seed"] == pair["seed"] and r["engine_steps"] == 3001
            assert len(r["decisions"]) == 30 and sum(r["action_counts_compact_default_stretched"]) == 30
            assert r["all_left_actions_builtin_ai"] and r["right_external_action_slots"] == 0
            assert r["opponent_tactical_setter_calls_after_reset"] == 0
            assert all(abs(d["live_native_depth_end"] - 0.71) < 1e-3 for d in r["decisions"])
            assert all(abs(d["live_coached_depth_end"] - {0: 0.0, 1: 0.71, 2: 1.0}[d["action"]]) < 1e-3
                       for d in r["decisions"])
            assert r["rl_goal_difference"] == r["rl_goals"] - r["native_goals"]
            own, opp = (r["score_left_right"] if r["coached_side"] == "left"
                        else r["score_left_right"][::-1])
            assert (r["rl_goals"], r["native_goals"]) == (own, opp)
            assert r["result"] == ("win" if own > opp else "loss" if own < opp else "draw")
            assert r["action_counts_compact_default_stretched"] == [
                sum(d["action"] == action for d in r["decisions"]) for action in range(3)]
            rows.append(r)
    n = len(seeds)
    gd = np.array([[next(r["rl_goal_difference"] for r in p["legs"] if r["coached_side"] == s)
                    for s in ("left", "right")] for p in pairs], float)
    gf = np.array([[next(r["rl_goals"] for r in p["legs"] if r["coached_side"] == s)
                    for s in ("left", "right")] for p in pairs], float)
    ga = np.array([[next(r["native_goals"] for r in p["legs"] if r["coached_side"] == s)
                    for s in ("left", "right")] for p in pairs], float)
    rng = np.random.default_rng(961234)
    idx = rng.integers(0, n, (20000, n))
    paired_mean = gd.mean(axis=1)
    boot = paired_mean[idx].mean(axis=1)
    summary = {"protocol": protocol, "n_seed_pairs": n, "n_matches": len(rows),
               "overall": {"wins": sum(r["result"] == "win" for r in rows),
                           "draws": sum(r["result"] == "draw" for r in rows),
                           "losses": sum(r["result"] == "loss" for r in rows),
                           "rl_goals_per_match": float(gf.mean()),
                           "native_goals_per_match": float(ga.mean()),
                           "rl_goal_difference_per_match": float(paired_mean.mean()),
                           "rl_goal_difference_ci95_seed_pair_bootstrap": ci(boot),
                           "rl_goal_difference_sum": int(gd.sum())},
               "by_coached_side": {}, "side_difference": {}, "action_counts": {},
               "bootstrap_seed": 961234, "bootstrap_resamples": 20000}
    for j, side in enumerate(("left", "right")):
        rs = [r for r in rows if r["coached_side"] == side]
        summary["by_coached_side"][side] = {
            "n": len(rs), "wins": sum(r["result"] == "win" for r in rs),
            "draws": sum(r["result"] == "draw" for r in rs),
            "losses": sum(r["result"] == "loss" for r in rs),
            "rl_goals_per_match": float(gf[:, j].mean()),
            "native_goals_per_match": float(ga[:, j].mean()),
            "rl_goal_difference_per_match": float(gd[:, j].mean()),
            "rl_goal_difference_ci95_seed_pair_bootstrap": ci(gd[idx, j].mean(axis=1))}
    diff = gd[:, 0] - gd[:, 1]
    summary["side_difference"] = {"rl_left_minus_rl_right_gd_per_match": float(diff.mean()),
                                  "ci95_seed_pair_bootstrap": ci(diff[idx].mean(axis=1))}
    for side in ("overall", "left", "right"):
        subset = rows if side == "overall" else [r for r in rows if r["coached_side"] == side]
        counts = np.sum([r["action_counts_compact_default_stretched"] for r in subset], axis=0)
        summary["action_counts"][side] = {"compact_default_stretched": counts.astype(int).tolist(),
                                          "fractions": (counts / counts.sum()).tolist(),
                                          "n_decisions": int(counts.sum())}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (folder / "matches.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "coached_side", "rl_goals", "native_goals", "rl_goal_difference",
                     "result", "compact_decisions", "default_decisions", "stretched_decisions"])
        for r in sorted(rows, key=lambda z: (z["seed"], z["coached_side"])):
            wr.writerow([r["seed"], r["coached_side"], r["rl_goals"], r["native_goals"],
                         r["rl_goal_difference"], r["result"]] + r["action_counts_compact_default_stretched"])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, side in enumerate(("left", "right", "overall")):
        z = summary["overall"] if side == "overall" else summary["by_coached_side"][side]
        m = z["rl_goal_difference_per_match"]
        lo, hi = z["rl_goal_difference_ci95_seed_pair_bootstrap"]
        axs[0].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", capsize=4,
                        color="#176b48" if side == "overall" else "#555555")
    axs[0].axhline(0, color="black", lw=1)
    axs[0].set_xticks(range(3), ["RL left", "RL right", "Both sides"])
    axs[0].set_ylabel("RL goal difference per match")
    axs[0].set_title("20,000 seed-pair bootstrap samples")
    for i, label in enumerate(("wins", "draws", "losses")):
        vals = [summary["by_coached_side"][s][label] for s in ("left", "right")]
        axs[1].bar(np.array([0, 1]) + (i - 1) * 0.23, vals, width=0.22,
                   color=("#176b48", "#a0a0a0", "#b34a3a")[i], label=label.title())
    axs[1].set_xticks([0, 1], ["RL left", "RL right"])
    axs[1].set_ylabel("Matches")
    axs[1].set_title("Direct Q versus native results")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "head_to_head_summary.png", dpi=170)

    c = summary["overall"]["rl_goal_difference_ci95_seed_pair_bootstrap"]
    verdict = ("Q outperformed native on goal difference in this sample and its 95% paired interval excludes zero."
               if c[0] > 0 else
               "Q underperformed native on goal difference in this sample and its 95% paired interval excludes zero."
               if c[1] < 0 else
               "The 95% paired interval includes zero; this sample does not establish a head-to-head advantage for either side.")
    report = ["# GRF frozen Q-learning coach versus native AI", "",
              "This is a direct match: Q changes only its own team's native defensive-depth parameter every 100 engine steps. All 11 exposed left-side players receive `action_builtin_ai` on every engine step; the right side has no external action slots. The opposing team has its override cleared at reset and is never changed during play. Both teams retain GRF's built-in passing, movement, tackling and shooting.", "",
              "The saved iteration-100 Q checkpoint is unchanged. The existing training feature builder, state discretizer and greedy action rule are used for either side. For Q on the right, GRF's official 180-degree observation rotation converts ball, teams, possession and score to the left-to-right perspective used in training.", "",
              "The interface is grounded in `coach_env.py` (`apply_depth`, `_features`, `discretize`), the existing two-team C++ binding in `grf_coach_probe/native_depth_poc/apply_patch.py`, and GRF's `gfootball/env/observation_rotation.py`. `head_to_head_probe.py` verified at runtime that setting team 1 leaves team 0's live depth at native default, that clearing restores team 1's native default, and that GRF's official rotation produces the right-side training perspective. No engine or training source was changed for this evaluation.", "",
              f"**Design:** {n} predetermined seeds ({seeds[0]}-{seeds[-1]}), two full 3,001-step matches per seed (Q left and Q right), {len(rows)} matches total. Leg order alternates by seed parity. The same scenario and seed are used for both legs; cross-process GRF nondeterminism means they are comparable, not guaranteed frame-identical mirrors. No seeds were chosen by outcome. Demonstration seed 970001 is excluded.", "",
              "| Q side | Matches | W/D/L | RL goals/match | Native goals/match | RL goal diff/match [95% CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for side in ("left", "right", "overall"):
        z = summary["overall"] if side == "overall" else summary["by_coached_side"][side]
        label = "Both sides" if side == "overall" else "RL " + side
        report.append("| %s | %d | %d/%d/%d | %.3f | %.3f | %+.3f [%+.3f, %+.3f] |" %
                      (label, len(rows) if side == "overall" else z["n"], z["wins"], z["draws"], z["losses"],
                       z["rl_goals_per_match"], z["native_goals_per_match"], z["rl_goal_difference_per_match"],
                       *z["rl_goal_difference_ci95_seed_pair_bootstrap"]))
    report += ["", "Intervals resample complete two-leg seed pairs (20,000 bootstrap samples), so the two side-swapped matches for a seed remain together. The RL-left minus RL-right goal-difference contrast is %+.3f [95%% CI %+.3f, %+.3f] goals/match." %
               (summary["side_difference"]["rl_left_minus_rl_right_gd_per_match"],
                *summary["side_difference"]["ci95_seed_pair_bootstrap"]), "",
               "| Q side | Decisions | Compact | Native default | Stretched |",
               "|---|---:|---:|---:|---:|"]
    for side in ("left", "right", "overall"):
        z = summary["action_counts"][side]
        report.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% |" %
                      ("Both sides" if side == "overall" else "RL " + side, z["n_decisions"],
                       *(100 * x for x in z["fractions"])))
    report += ["", "**Conclusion:** " + verdict,
               "The +0.480 [95% CI +0.100, +0.860] RL-left versus RL-right contrast shows material side sensitivity. Pooling the two legs balances this for the main estimate. The contrast alone cannot distinguish an intrinsic left/right scenario advantage from a right-side transfer weakness in Q; no native-versus-native side baseline was run on these fresh seeds. The Q-left point estimate is therefore not evidence of a general advantage.", "",
               "The Q-left leg is a fresh-seed replication of the earlier Q-versus-native setup; the Q-right leg adds the side swap. The direct matchup does not by itself show that Q learned useful state dependence: the observation-shuffling control in the earlier held-out evaluation found no goal-difference gain.", "",
               "Raw decisions for every match are in `seed_*.json`; `matches.csv`, `summary.json` and `head_to_head_summary.png` provide compact views."]
    demo_file = Path("demo/head_to_head_seed970001/summary.json")
    if demo_file.exists():
        demo = json.loads(demo_file.read_text())
        assert demo["seed"] not in seeds and demo["checkpoint_sha256"] == protocol["checkpoint_sha256"]
        results = {x["coached_side"]: x["rl_goal_difference"] for x in demo["legs"]}
        report += ["", "**Separate demonstrations:** The predetermined seed 970001 gave RL goal difference %+.0f on the left and %+.0f on the right. These two gameplay recordings and per-decision timelines are in `../../demo/head_to_head_seed970001/` and are excluded from the aggregate estimates." %
                   (results["left"], results["right"])]
    report += ["",
               "Reproduce from `grf_rl_coach/`:", "", "```bash",
               "bash run.sh head_to_head_probe.py",
               "bash run.sh head_to_head_eval.py --seeds 100 --seed-start 960000 --workers 6 --out results/head_to_head_eval",
               "python analyze_head_to_head.py --folder results/head_to_head_eval",
               "bash run.sh head_to_head_demo.py --seed 970001 --out demo/head_to_head_reproduce_seed970001",
               "```", ""]
    (folder / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(verdict)
    print("W/D/L", summary["overall"]["wins"], summary["overall"]["draws"], summary["overall"]["losses"])
    print("GD", summary["overall"]["rl_goal_difference_per_match"], c)
    print("written", folder / "REPORT.md")


if __name__ == "__main__":
    main()

"""PRELIMINARY report from a head-to-head evaluation folder: fixed-left comparison (left legs only) and two-sided evaluation (both legs).
usage: bash run.sh prelim_report.py [results_folder_name] [out_tag]      default folder eval_h2h_prelim, tag _prelim
Both tables use overlapping match data: the fixed-left matches ARE the left legs of the two-sided run (same seeds, same engine, deterministic).
Intervals: 95% percentile bootstrap over seeds (fixed-left) or seed pairs (two-sided); p-values: sign-flip permutation test (unadjusted)."""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "eval_h2h_prelim"
TAG = sys.argv[2] if len(sys.argv) > 2 else "_prelim"
R = Path("results"); B = 10000; rng = np.random.default_rng(0)
NAMES = {"native": "Native GRF", "A_q_original": "A  Q-learning, original reward", "B_q_possession": "B  Q-learning, possession reward",
         "C_dq_possession": "C  Double Q, possession reward"}
ORDER = ["native", "A_q_original", "B_q_possession", "C_dq_possession"]


def ci(x):
    x = np.asarray(x, float); bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def fmt(x, c, p=2, sign=True):
    s = "{:+.%df}" % p if sign else "{:.%df}" % p
    return s.format(x) + " [" + s.format(c[0]) + ", " + s.format(c[1]) + "]"


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); c = ci(d)
    sg = rng.choice([-1.0, 1.0], (B, len(d))); p = float(((np.abs((sg * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    return d.mean(), c, p


by = {}
for f in sorted(glob.glob(str(R / FOLDER / "*.pkl"))):
    lab = Path(f).stem
    by[lab] = {(ep["seed"], ep["side"]): ep for ep, _ in pickle.load(open(f, "rb"))}
labels = [l for l in ORDER if l in by]
seeds = sorted({s for l in labels for s, sd in by[l] if sd == 0})
for l in labels:
    assert all((s, 0) in by[l] and (s, 1) in by[l] for s in seeds), f"{l} is missing legs"
n = len(seeds)


def leg(l, side, key): return np.array([by[l][(s, side)][key] for s in seeds], float)


def wdl(gd):
    w, d, lo = int((gd > 0).sum()), int((gd == 0).sum()), int((gd < 0).sum()); m = len(gd)
    return w, d, lo, f"{w}-{d}-{lo} ({100*w/m:.0f}/{100*d/m:.0f}/{100*lo/m:.0f}%)"


out, js = [], {}
out.append(f"# PRELIMINARY evaluation: {n} fresh seed pairs, seeds {seeds[0]:,} to {seeds[-1]:,} (folder {FOLDER})\n")
out.append("Final iteration-100 policies, greedy (no exploration). Preliminary: small sample, intervals are wide; not for claims of improvement unless an interval excludes zero.\n")
out.append("The two tables below use OVERLAPPING match data: the fixed-left matches are the left legs of the two-sided run (same seeds, deterministic engine).\n")

# ------------------------------------------------------------ fixed-left
out.append(f"\n## Table 1. Fixed-left comparison ({n} matches per policy; policy coaches the LEFT team, opponent = unmodified native AI on the right)\n")
out.append("| Policy | W-D-L (win/draw/loss %) | Goals for | Goals against | Mean goal diff [95% CI] | Possession % [95% CI] | Matches >50% possession |")
out.append("|---|---|---|---|---|---|---|")
fl = {}
for l in labels:
    gd = leg(l, 0, "goal_diff"); ps = 100 * leg(l, 0, "possession_share"); fl[l] = (gd, ps)
    out.append(f"| {NAMES[l]} | {wdl(gd)[3]} | {leg(l,0,'goals_for').mean():.2f} | {leg(l,0,'goals_against').mean():.2f} | {fmt(gd.mean(), ci(gd))} | "
               f"{fmt(ps.mean(), ci(ps), 1, False)} | {int((ps > 50).sum())}/{n} |")
    js.setdefault("fixed_left", {})[l] = dict(wdl=wdl(gd)[:3], gd=float(gd.mean()), possession=float(ps.mean()))
out.append("\nPaired differences, same seeds (positive = first policy higher), 95% bootstrap CI over seeds, unadjusted sign-flip p:\n")
out.append("| Comparison | Goal diff difference [95% CI] | p | Possession difference, pts [95% CI] | p |")
out.append("|---|---|---|---|---|")
pairs = [(a, b) for a, b in [("A_q_original", "native"), ("B_q_possession", "native"), ("C_dq_possession", "native"), ("B_q_possession", "A_q_original"),
                             ("C_dq_possession", "A_q_original"), ("C_dq_possession", "B_q_possession")] if a in fl and b in fl]
for a, b in pairs:
    g = paired(fl[a][0], fl[b][0]); p = paired(fl[a][1], fl[b][1])
    out.append(f"| {NAMES[a][0] if a != 'native' else 'Native'} vs {NAMES[b][0] if b != 'native' else 'Native'} | {fmt(g[0], g[1])} | {g[2]:.3f} | {fmt(p[0], p[1], 1)} | {p[2]:.3f} |")
    js.setdefault("fixed_left_paired", {})[f"{a}_vs_{b}"] = dict(gd=g[0], gd_ci=g[1], gd_p=g[2], poss=p[0], poss_ci=p[1], poss_p=p[2])
out.append("\nNote: the right team has slightly weaker players in GRF (difficulty 0.95 vs 1.0), so the left side has a built-in advantage; fixed-left means all coached teams share it.")

# ------------------------------------------------------------ two-sided
out.append(f"\n## Table 2. Two-sided evaluation ({n} seed pairs = {2*n} matches per policy; policy coaches the left team in one leg and the right team in the other)\n")
out.append("| Policy | Legs W-D-L (win/draw/loss %) | Left leg GD | Right leg GD | Pair-mean goal diff [95% CI] | Pair-mean possession % [95% CI] | Seed pairs won-drawn-lost (by combined GD) |")
out.append("|---|---|---|---|---|---|---|")
ts = {}
for l in labels:
    gl, gr = leg(l, 0, "goal_diff"), leg(l, 1, "goal_diff"); pl, pr = 100 * leg(l, 0, "possession_share"), 100 * leg(l, 1, "possession_share")
    pg, pp = (gl + gr) / 2, (pl + pr) / 2; ts[l] = (pg, pp)
    legs = np.concatenate([gl, gr]); comb = gl + gr
    out.append(f"| {NAMES[l]} | {wdl(legs)[3]} | {gl.mean():+.2f} | {gr.mean():+.2f} | {fmt(pg.mean(), ci(pg))} | {fmt(pp.mean(), ci(pp), 1, False)} | "
               f"{int((comb > 0).sum())}-{int((comb == 0).sum())}-{int((comb < 0).sum())} |")
    js.setdefault("two_sided", {})[l] = dict(wdl=wdl(legs)[:3], gd_left=float(gl.mean()), gd_right=float(gr.mean()), pair_gd=float(pg.mean()), pair_possession=float(pp.mean()))
out.append("\nPaired differences over seed pairs (pair-mean values; positive = first policy higher), 95% bootstrap CI, unadjusted sign-flip p:\n")
out.append("| Comparison | Pair-mean goal diff difference [95% CI] | p | Possession difference, pts [95% CI] | p |")
out.append("|---|---|---|---|---|")
for a, b in pairs:
    g = paired(ts[a][0], ts[b][0]); p = paired(ts[a][1], ts[b][1])
    out.append(f"| {NAMES[a][0] if a != 'native' else 'Native'} vs {NAMES[b][0] if b != 'native' else 'Native'} | {fmt(g[0], g[1])} | {g[2]:.3f} | {fmt(p[0], p[1], 1)} | {p[2]:.3f} |")
    js.setdefault("two_sided_paired", {})[f"{a}_vs_{b}"] = dict(gd=g[0], gd_ci=g[1], gd_p=g[2], poss=p[0], poss_ci=p[1], poss_p=p[2])
out.append("\nNative vs native: both legs are the same match seen from the two sides, so its pair-mean goal difference is exactly 0 and its pair-mean possession exactly 50%; "
           "the two-sided differences against native therefore equal each policy's own pair mean. Left minus right leg shows side sensitivity.")
out.append("\nNo multiplicity adjustment is applied in this preliminary report; with 6 comparisons x 2 metrics x 2 tables, a few small p-values are expected by chance.")
Path(f"results/prelim_tables{TAG}.md").write_text("\n".join(out), encoding="utf-8")
json.dump(js, open(f"results/prelim_summary{TAG}.json", "w"), indent=1, default=float)
print("\n".join(out))

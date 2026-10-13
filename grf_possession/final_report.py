"""Final 800-match evaluation report: native, A, B, C x 100 seed pairs x both sides (folder results/eval_final100).
usage: bash run.sh final_report.py [folder] [tag]
Outputs results/final100_tables.md and results/final100_summary.json (all tables, every comparison reported, no selection).
Intervals: 95% percentile bootstrap over seeds (fixed-left, one side) or seed pairs (two-sided); p: sign-flip permutation (unadjusted);
Holm adjustment over the pre-declared primary family {B vs native, C vs native} x {goal difference, possession} (two-sided and fixed-left separately)."""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "eval_final100"
TAG = sys.argv[2] if len(sys.argv) > 2 else "final100"
R = Path("results"); B = 10000; rng = np.random.default_rng(0)
ORDER = ["native", "A_q_original", "B_q_possession", "C_dq_possession"]
SH = {"native": "Native GRF", "A_q_original": "Q-learning A", "B_q_possession": "Q-learning B", "C_dq_possession": "Double Q-learning C"}
FULL = {"native": "Native GRF", "A_q_original": "A (Q, original reward)", "B_q_possession": "B (Q, possession reward)", "C_dq_possession": "C (Double Q, possession reward)"}


def ci(x):
    x = np.asarray(x, float); bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); c = ci(d)
    sg = rng.choice([-1.0, 1.0], (B, len(d))); p = float(((np.abs((sg * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    return float(d.mean()), c, p


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def f2(x, c, p=2, sign=True):
    s = "{:+.%df}" % p if sign else "{:.%df}" % p
    return s.format(x) + " [" + s.format(c[0]) + ", " + s.format(c[1]) + "]"


by = {}
for f in sorted(glob.glob(str(R / FOLDER / "*.pkl"))):
    by[Path(f).stem] = {(ep["seed"], ep["side"]): ep for ep, _ in pickle.load(open(f, "rb"))}
labels = [l for l in ORDER if l in by]
seeds = sorted({s for l in labels for s, sd in by[l] if sd == 0}); n = len(seeds)
for l in labels: assert all((s, sd) in by[l] for s in seeds for sd in (0, 1)), f"{l} incomplete"


def leg(l, side, key): return np.array([by[l][(s, side)][key] for s in seeds], float)


def counts(gd): return int((gd > 0).sum()), int((gd == 0).sum()), int((gd < 0).sum())


def stat_block(l, sides):
    gd = np.concatenate([leg(l, sd, "goal_diff") for sd in sides]); gf = np.concatenate([leg(l, sd, "goals_for") for sd in sides])
    ga = np.concatenate([leg(l, sd, "goals_against") for sd in sides]); ps = 100 * np.concatenate([leg(l, sd, "possession_share") for sd in sides])
    w, d, lo = counts(gd); m = len(gd)
    return dict(m=m, w=w, d=d, l=lo, win=100 * w / m, loss=100 * lo / m, gf=gf.mean(), ga=ga.mean(), gd=gd.mean(), gd_ci=ci(gd), poss=ps.mean(), poss_ci=ci(ps),
                gt50=int((ps > 50).sum()))


js, out = {}, []
out.append(f"# Final evaluation: {n} fresh seed pairs, seeds {seeds[0]:,}-{seeds[-1]:,}, every policy on both sides = {2*n} matches per policy, {2*n*len(labels)} in total\n")
out.append("Frozen iteration-100 checkpoints, greedy, no exploration, opponent = unmodified native GRF AI. Same seeds, scenario and settings for every policy.\n")
out.append("Fixed-left and two-sided results use OVERLAPPING match data (the fixed-left matches are the left legs of the two-sided matches). Native vs native: both legs are the same match seen from the two sides, so its two-sided figures are exactly neutral (goal difference 0, possession 50%).\n")


def format_table(title, sides, note=""):
    S = {l: stat_block(l, sides) for l in labels}; js[title] = {l: {k: (list(v) if isinstance(v, tuple) else v) for k, v in S[l].items()} for l in labels}
    out.append(f"\n## {title}\n")
    if note: out.append(note + "\n")
    out.append("| Result | " + " | ".join(SH[l] for l in labels) + " |"); out.append("|---|" + "---|" * len(labels))
    rows = [("Matches", lambda s: f"{s['m']}"), ("Wins", lambda s: f"{s['w']}"), ("Draws", lambda s: f"{s['d']}"), ("Losses", lambda s: f"{s['l']}"),
            ("Win percentage", lambda s: f"{s['win']:.1f}%"), ("Loss percentage", lambda s: f"{s['loss']:.1f}%"), ("Goals per match", lambda s: f"{s['gf']:.2f}"),
            ("Goals conceded per match", lambda s: f"{s['ga']:.2f}"), ("Average goal difference", lambda s: f"{s['gd']:+.2f}"),
            ("  95% CI (goal difference)", lambda s: f"[{s['gd_ci'][0]:+.2f}, {s['gd_ci'][1]:+.2f}]"),
            ("Average possession percentage", lambda s: f"{s['poss']:.1f}%"), ("  95% CI (possession)", lambda s: f"[{s['poss_ci'][0]:.1f}, {s['poss_ci'][1]:.1f}]"),
            ("Matches with >50% possession", lambda s: f"{s['gt50']}/{s['m']}")]
    for name, fn in rows: out.append(f"| {name} | " + " | ".join(fn(S[l]) for l in labels) + " |")


def h2h_table(title, sides):
    out.append(f"\n## {title}\n")
    out.append("| Result | " + " | ".join(f"{SH[l]} | Native opponent" for l in labels if l != "native") + " |")
    out.append("|---|" + "---|---|" * (len(labels) - 1))
    S = {l: stat_block(l, sides) for l in labels if l != "native"}
    def row(name, fa, fb): out.append(f"| {name} | " + " | ".join(f"{fa(S[l])} | {fb(S[l])}" for l in S) + " |")
    row("Wins", lambda s: f"{s['w']} ({s['win']:.1f}%)", lambda s: f"{s['l']} ({s['loss']:.1f}%)")
    row("Draws", lambda s: f"{s['d']} ({100*s['d']/s['m']:.1f}%)", lambda s: f"{s['d']} ({100*s['d']/s['m']:.1f}%)")
    row("Losses", lambda s: f"{s['l']} ({s['loss']:.1f}%)", lambda s: f"{s['w']} ({s['win']:.1f}%)")
    row("Goals per match", lambda s: f"{s['gf']:.2f}", lambda s: f"{s['ga']:.2f}")
    row("Goal difference", lambda s: f"{s['gd']:+.2f}", lambda s: f"{-s['gd']:+.2f}")
    row("Possession %", lambda s: f"{s['poss']:.1f}", lambda s: f"{100 - s['poss']:.1f}")


format_table(f"Table 1. FIXED-LEFT: each policy coaches the LEFT team ({n} matches each) against unmodified native GRF on the right", (0,),
             "Left-side players are slightly stronger in GRF (difficulty 1.0 vs 0.95), so all left-side results include a built-in left advantage.")
h2h_table(f"Table 1b. FIXED-LEFT head-to-head, coach on the left vs native on the right ({n} matches per policy)", (0,))
format_table(f"Table 2a. Coach on the RIGHT ({n} matches each) against native GRF on the left", (1,), "Right side is the disadvantaged side in this scenario.")
h2h_table(f"Table 2b. RIGHT-side head-to-head, coach on the right vs native on the left ({n} matches per policy)", (1,))
format_table(f"Table 3. COMBINED left + right ({2*n} matches per policy)", (0, 1), "Native vs native: the two legs are one match seen from both sides, so every native row is exactly neutral.")
h2h_table(f"Table 3b. COMBINED head-to-head (left + right, {2*n} matches per policy)", (0, 1))

# ---------------------------------------------------------------- paired statistics
def metric_arrays(l, view):
    if view == "fixed_left": return {"gd": leg(l, 0, "goal_diff"), "poss": 100 * leg(l, 0, "possession_share"), "win": 100 * (leg(l, 0, "goal_diff") > 0), "loss": 100 * (leg(l, 0, "goal_diff") < 0)}
    if view == "right_only": return {"gd": leg(l, 1, "goal_diff"), "poss": 100 * leg(l, 1, "possession_share"), "win": 100 * (leg(l, 1, "goal_diff") > 0), "loss": 100 * (leg(l, 1, "goal_diff") < 0)}
    gl, gr = leg(l, 0, "goal_diff"), leg(l, 1, "goal_diff")                       # two-sided: pair means over seed pairs
    return {"gd": (gl + gr) / 2, "poss": 50 * (leg(l, 0, "possession_share") + leg(l, 1, "possession_share")) , "win": 50 * ((gl > 0) * 1.0 + (gr > 0)), "loss": 50 * ((gl < 0) * 1.0 + (gr < 0))}


COMP = [("B_q_possession", "native"), ("C_dq_possession", "native"), ("B_q_possession", "A_q_original"), ("C_dq_possession", "B_q_possession"),
        ("A_q_original", "native"), ("C_dq_possession", "A_q_original")]
PRIMARY = [("B_q_possession", "native"), ("C_dq_possession", "native")]
for view, vt in [("fixed_left", f"Table 4. PAIRED comparisons, FIXED-LEFT ({n} matched seeds)"), ("two_sided", f"Table 5. PAIRED comparisons, TWO-SIDED (pair-mean over {n} seed pairs)"),
                 ("right_only", f"Table 6. PAIRED comparisons, RIGHT side only ({n} matched seeds)")]:
    M = {l: metric_arrays(l, view) for l in labels}; res = {}
    out.append(f"\n## {vt}\n")
    out.append("Positive = first policy higher. Win/loss differences are in percentage points. p = unadjusted sign-flip permutation p; Holm column covers the primary family (B/C vs native, goal difference and possession).\n")
    out.append("| Comparison | Goal diff difference [95% CI] | p | Possession pts [95% CI] | p | Win-rate pts [95% CI] | Loss-rate pts [95% CI] |"); out.append("|---|---|---|---|---|---|---|")
    for a, b in COMP:
        if a not in M or b not in M: continue
        g, p_, w, lo = paired(M[a]["gd"], M[b]["gd"]), paired(M[a]["poss"], M[b]["poss"]), paired(M[a]["win"], M[b]["win"]), paired(M[a]["loss"], M[b]["loss"])
        res[f"{a}_vs_{b}"] = dict(gd=g, poss=p_, win=w, loss=lo)
        tag = " (primary)" if (a, b) in PRIMARY else ""
        out.append(f"| {SH[a]} vs {SH[b]}{tag} | {f2(g[0], g[1])} | {g[2]:.3f} | {f2(p_[0], p_[1], 1)} | {p_[2]:.3f} | {f2(w[0], w[1], 1)} | {f2(lo[0], lo[1], 1)} |")
    fam = [res[f"{a}_vs_{b}"][m][2] for a, b in PRIMARY if f"{a}_vs_{b}" in res for m in ("gd", "poss")]
    if fam:
        adj = holm(fam); names = [f"{SH[a]} vs {SH[b]} / {m}" for a, b in PRIMARY if f"{a}_vs_{b}" in res for m in ("goal diff", "possession")]
        out.append("\nHolm-adjusted p (primary family): " + "; ".join(f"{nm} {x:.3f}" for nm, x in zip(names, adj)))
        res["holm_primary"] = dict(zip(names, map(float, adj)))
    js[vt] = res

out.append(f"\n## Reading guide\n\n- An improvement is called reliable only if the paired 95% interval excludes zero (and, for the primary family, the Holm-adjusted p is below 0.05). All comparisons are shown, favourable or not.\n"
           "- Possession is part of the training reward of B and C (not A or native), so a possession gain for them is the trained-for target; goal difference and win/loss rates are the outcome measures.\n"
           "- Side sensitivity: compare Tables 1 (left) and 2a (right). Two-sided pair means cancel the built-in left-side advantage.\n")
Path(f"results/{TAG}_tables.md").write_text("\n".join(out), encoding="utf-8")
json.dump(js, open(f"results/{TAG}_summary.json", "w"), indent=1, default=float)
print("\n".join(out))

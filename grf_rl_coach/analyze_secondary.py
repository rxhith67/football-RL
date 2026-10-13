"""Exploratory secondary analysis of football statistics (possession, recoveries, opponent advancement, shots, goals).
usage: bash run.sh analyze_secondary.py            (reads results/secondary/*.pkl, writes results/secondary_* and figs/secondary_*)
Paired by seed (identical seeds across policies). Intervals are 95% percentile bootstrap over matches; p-values are
paired sign-flip permutation tests; Holm adjustment is applied across the Q-learning-vs-native family."""
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

R = Path("results"); S = R / "secondary"
ORDER = ["native_unmodified", "q_learning", "fixed_compact", "fixed_stretched", "hand_coach", "random_switching", "q_learning_shuffled_obs"]
NICE = {"native_unmodified": "Native", "q_learning": "Q-learning", "fixed_compact": "Always compact", "fixed_stretched": "Always stretched",
        "hand_coach": "Hand-written", "random_switching": "Random switching", "q_learning_shuffled_obs": "Q, shuffled obs."}
COL = {"native_unmodified": "#555555", "q_learning": "#1b6b45", "fixed_compact": "#3b82f6", "fixed_stretched": "#f97316",
       "hand_coach": "#8b5cf6", "random_switching": "#aaaaaa", "q_learning_shuffled_obs": "#7cc39a"}
# metric: (label, +1 if higher is better / -1 if lower is better)
MET = {"possession_share_debounced": ("Possession share (debounced)", +1), "possession_share_owned": ("Possession share (owned steps)", +1),
       "recoveries_live": ("Live recoveries / match", +1), "recoveries_live_per_1000_opp_steps": ("Recoveries per 1000 opp. steps", +1),
       "possession_losses": ("Possession losses / match", -1), "time_to_recover_rmst": ("Time to recover, RMST (steps)", -1),
       "time_to_recover_completed": ("Time to recover, completed only (steps)", -1), "opp_adv_mean": ("Opp. advancement, mean", -1),
       "opp_adv_max": ("Opp. advancement, max", -1), "opp_third_share": ("Opp. possession in our third", -1), "danger_steps": ("Danger-zone steps", -1),
       "shots_conceded": ("Shots conceded (heuristic)", -1), "shots_on_target_conceded": ("Shots on target conceded (heuristic)", -1),
       "goals_conceded": ("Goals conceded", -1), "goals_scored": ("Goals scored", +1), "shots_for": ("Shots taken (heuristic)", +1), "goal_diff": ("Goal difference", +1)}
KEY9 = ["possession_share_debounced", "recoveries_live", "time_to_recover_rmst", "opp_adv_mean", "danger_steps", "shots_conceded",
        "shots_on_target_conceded", "goals_conceded", "goal_diff"]
rng = np.random.default_rng(5)
B = 10000

raw = {l: pickle.load(open(S / f"{l}.pkl", "rb")) for l in ORDER if (S / f"{l}.pkl").exists()}
labels = [l for l in ORDER if l in raw]
seeds = sorted(ep["seed"] for ep, _, _ in raw[labels[0]])
val = {}
for l in labels:
    by = {ep["seed"]: (ep, st, dec) for ep, st, dec in raw[l]}
    ms = [M.match_metrics(by[s][1]) for s in seeds]
    val[l] = {k: np.array([m[k] for m in ms], float) for k in ms[0]}
    val[l]["_ep"] = [by[s][0] for s in seeds]; val[l]["_dec"] = [by[s][2] for s in seeds]; val[l]["_st"] = [by[s][1] for s in seeds]
n = len(seeds)

def se(x):
    x = x[~np.isnan(x)]; return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan

def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); d = (a - b)[ok]
    if len(d) < 5: return dict(diff=np.nan, ci=[np.nan, np.nan], p=np.nan, dz=np.nan, n=int(ok.sum()))
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    signs = rng.choice([-1.0, 1.0], (B, len(d)))
    p = float(((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    sd = d.std(ddof=1)
    return dict(diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], p=p, dz=float(d.mean() / sd) if sd > 0 else 0.0, n=int(ok.sum()))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

out = {"n_matches_per_policy": n, "seeds": [seeds[0], seeds[-1]], "summary": {}, "paired": {}}
for l in labels:
    out["summary"][l] = {k: [float(np.nanmean(val[l][k])), se(val[l][k])] for k in MET if k in val[l]}
    ac = np.sum([e["action_counts"] for e in val[l]["_ep"]], 0); out["summary"][l]["action_fraction"] = (ac / ac.sum()).round(3).tolist()
comps = [("q_learning", "native_unmodified"), ("fixed_compact", "native_unmodified"), ("fixed_stretched", "native_unmodified"),
         ("hand_coach", "native_unmodified"), ("random_switching", "native_unmodified"),
         ("q_learning", "fixed_compact"), ("q_learning", "fixed_stretched"), ("q_learning", "hand_coach"), ("q_learning", "random_switching"),
         ("q_learning", "q_learning_shuffled_obs")]
for a, b in comps:
    if a in val and b in val:
        out["paired"][f"{a} minus {b}"] = {k: paired(val[a][k], val[b][k]) for k in MET}
fam = out["paired"]["q_learning minus native_unmodified"]
adj = holm([fam[k]["p"] for k in MET]);
for k, a in zip(MET, adj): fam[k]["p_holm"] = float(a)

# ---------- situation dependence ----------
def situation(dec):
    f = dec[:, :14]
    zone = np.digitize(f[:, 0], [-0.33, 0.33]); poss = np.argmax(f[:, 2:5], 1)
    sd = f[:, 5] * 3.0; score = np.digitize(sd, [-0.5, 0.5]); t = f[:, 6]; phase = np.where(t > 2 / 3, 0, np.where(t > 1 / 3, 1, 2))
    return {"ball_zone": zone, "possession": poss, "score_state": score, "match_phase": phase}, dec[:, 14].astype(int)

def mutual_info(x, a):
    mi = 0.0
    for xv in np.unique(x):
        for av in range(3):
            pxa = np.mean((x == xv) & (a == av)); px = np.mean(x == xv); pa = np.mean(a == av)
            if pxa > 0: mi += pxa * np.log2(pxa / (px * pa))
    return mi

dep = {}
for l in ("q_learning", "q_learning_shuffled_obs", "random_switching", "hand_coach"):
    if l not in val: continue
    dec = np.concatenate(val[l]["_dec"]); dims, a = situation(dec)
    dep[l] = {}
    for dname, x in dims.items():
        mi = mutual_info(x, a)
        null = np.array([mutual_info(x, rng.permutation(a)) for _ in range(300)])
        dep[l][dname] = {"mutual_info_bits": float(mi), "null_mean": float(null.mean()), "p_perm": float((np.sum(null >= mi) + 1) / 301)}
    dep[l]["decisions"] = int(len(a))
out["action_situation_dependence"] = dep

# direct-method check: do the actions Q chose in each situation correspond to better short-term DEFENSIVE outcomes in
# RANDOMISED data (random switching)? interval outcome = danger-zone steps in the 100-step interval.
def interval_outcomes(l):
    rows = []
    for st, dec in zip(val[l]["_st"], val[l]["_dec"]):
        nd = len(dec)
        edges = [i * 100 for i in range(nd)] + [len(st)]
        for k in range(nd):
            seg = st[edges[k]:edges[k + 1]] if k < nd - 1 else st[edges[k]:]
            danger = ((seg[:, 1] < -0.76) & (np.abs(seg[:, 2]) < 0.22) & (seg[:, 0] != 0)).sum()
            rows.append(danger)
    return np.array(rows, float)
if "random_switching" in val:
    d_r = np.concatenate(val["random_switching"]["_dec"]); dims_r, a_r = situation(d_r); y_r = interval_outcomes("random_switching")
    cell_r = dims_r["ball_zone"] * 3 + dims_r["possession"]
    table = np.full((9, 3), np.nan); cnt = np.zeros((9, 3), int)
    for c in range(9):
        for a in range(3):
            m = (cell_r == c) & (a_r == a); cnt[c, a] = m.sum()
            if m.sum() >= 15: table[c, a] = y_r[m].mean()
    rel = table - np.nanmean(table, 1, keepdims=True)          # predicted danger steps relative to the cell average (negative = better)
    dm = {}
    for l in ("q_learning", "q_learning_shuffled_obs", "hand_coach", "random_switching"):
        if l not in val: continue
        dec = np.concatenate(val[l]["_dec"]); dims, a = situation(dec); cell = dims["ball_zone"] * 3 + dims["possession"]
        eff = np.array([rel[c, x] for c, x in zip(cell, a)]); ok = ~np.isnan(eff)
        # cluster bootstrap over matches
        per = np.split(eff, np.cumsum([len(d) for d in val[l]["_dec"]])[:-1])
        pm = np.array([np.nanmean(p) if np.any(~np.isnan(p)) else np.nan for p in per]); pm = pm[~np.isnan(pm)]
        bs = pm[rng.integers(0, len(pm), (B, len(pm)))].mean(1)
        dm[l] = {"mean_predicted_danger_steps_vs_cell_average": float(pm.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    out["direct_method_defensive_check"] = {"table_relative_danger_steps_by_cell_action[zone*3+possession][compact,default,stretched]": np.round(rel, 2).tolist(),
                                            "cell_action_counts": cnt.tolist(), "per_policy": dm}
json.dump(out, open(R / "secondary_analysis.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))

# ---------- tables ----------
L = [f"Exploratory secondary analysis, {n} matches per policy, seeds {seeds[0]} to {seeds[-1]} (identical seeds across policies). Mean ± SE.", "",
     "| Metric (better) | " + " | ".join(NICE[l] for l in labels) + " |", "|---|" + "---|" * len(labels)]
for k, (lab, sgn) in MET.items():
    row = [f"{out['summary'][l][k][0]:.3f} ± {out['summary'][l][k][1]:.3f}" for l in labels]
    L.append(f"| {lab} ({'higher' if sgn > 0 else 'lower'}) | " + " | ".join(row) + " |")
L += ["", "Paired difference vs native (policy minus native): mean [95% CI]; * = interval excludes 0 (unadjusted); markers: B better, W worse, - none.", "",
      "| Metric | " + " | ".join(NICE[l] for l in labels if l not in ("native_unmodified", "q_learning_shuffled_obs")) + " |", "|---|" + "---|" * (len(labels) - 2)]
for k, (lab, sgn) in MET.items():
    row = []
    for l in labels:
        if l in ("native_unmodified", "q_learning_shuffled_obs"): continue
        c = out["paired"][f"{l} minus native_unmodified"][k]
        sig = (c["ci"][0] > 0 or c["ci"][1] < 0)
        mark = "" if not sig else (" *B" if (c["diff"] * sgn > 0) else " *W")
        row.append(f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]{mark}")
    L.append(f"| {lab} | " + " | ".join(row) + " |")
L += ["", "Q-learning minus native with permutation p and Holm-adjusted p over all 17 metrics:", "", "| Metric | Difference [95% CI] | p | Holm p | Direction |", "|---|---|---|---|---|"]
for k, (lab, sgn) in MET.items():
    c = fam[k]
    L.append(f"| {lab} | {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['p']:.3f} | {c['p_holm']:.3f} | {('RELIABLY ' if c['p_holm'] < 0.05 else 'not reliable; point estimate ') + ('better' if c['diff'] * sgn > 0 else 'worse' if c['diff'] * sgn < 0 else 'equal')} |")
for comp in ("q_learning minus fixed_compact", "q_learning minus fixed_stretched", "q_learning minus q_learning_shuffled_obs"):
    if comp in out["paired"]:
        L += ["", f"{comp.replace('_', ' ')} (mean [95% CI], p):", ""]
        for k, (lab, sgn) in MET.items():
            c = out["paired"][comp][k]; L.append(f"- {lab}: {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], p = {c['p']:.3f}")
L += ["", "Action-situation dependence (mutual information between chosen action and situation, bits; permutation p):", ""]
for l, d in dep.items():
    L.append(f"- {NICE[l]} ({d['decisions']} decisions): " + ", ".join(f"{k} {v['mutual_info_bits']:.4f} (p = {v['p_perm']:.3f})" for k, v in d.items() if k != "decisions"))
if "direct_method_defensive_check" in out:
    L += ["", "Direct-method check (predicted change in danger-zone steps per interval from randomised data; negative = better):", ""]
    for l, v in out["direct_method_defensive_check"]["per_policy"].items():
        L.append(f"- {NICE[l]}: {v['mean_predicted_danger_steps_vs_cell_average']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
(R / "secondary_tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# ---------- figures ----------
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
Path("figs").mkdir(exist_ok=True)
fig, axs = plt.subplots(3, 3, figsize=(13, 8.5))
show = [l for l in labels if l != "q_learning_shuffled_obs"]
for ax, k in zip(axs.ravel(), KEY9):
    for i, l in enumerate(show):
        m, s = out["summary"][l][k]
        ax.errorbar(m, i, xerr=1.96 * s, fmt="o", color=COL[l], capsize=2.5)
    ax.axvline(out["summary"]["native_unmodified"][k][0], color="#c0392b", ls="--", lw=.8)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([NICE[l] for l in show] if ax in axs[:, 0] else [], fontsize=7.5); ax.invert_yaxis()
    ax.set_title(f"{MET[k][0]} ({'higher' if MET[k][1] > 0 else 'lower'} = better)", fontsize=8.5)
plt.suptitle(f"Football statistics by coach ({n} matches each, mean ± 1.96 SE; dashed = native)", y=1.0); plt.tight_layout()
plt.savefig("figs/secondary_overview.png", dpi=170)

fig, ax = plt.subplots(figsize=(8, 6)); ks = list(MET)
for i, k in enumerate(ks):
    c = fam[k]; d = c["dz"]; sgn = MET[k][1]
    ax.errorbar(d * sgn, i, xerr=None, fmt="o", color="#1b6b45" if d * sgn > 0 else "#c0392b")
    h = 1.96 / np.sqrt(c["n"])
    ax.plot([d * sgn - h, d * sgn + h], [i, i], color="#1b6b45" if d * sgn > 0 else "#c0392b", lw=1.2)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(ks))); ax.set_yticklabels([MET[k][0] for k in ks], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("standardised paired difference vs native\n(right = Q-learning better; approx. 95% interval)")
ax.set_title("Q-learning vs native on every measured statistic (exploratory)"); plt.tight_layout(); plt.savefig("figs/secondary_q_vs_native_effects.png", dpi=170)

fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
zn = ["own third", "middle", "opp. third"]; pn = ["we have ball", "opp. has ball", "loose"]
for ax, l in zip(axs[:2], ("q_learning", "random_switching")):
    dec = np.concatenate(val[l]["_dec"]); dims, a = situation(dec); cell = dims["ball_zone"] * 3 + dims["possession"]
    frac = np.array([[np.mean(a[cell == c] == x) if np.any(cell == c) else 0 for x in range(3)] for c in range(9)]); left = np.zeros(9)
    for x, (nm, cl) in enumerate((("compact", "#3b82f6"), ("native default", "#cbd5e1"), ("stretched", "#f97316"))):
        ax.bar(range(9), frac[:, x], bottom=left, color=cl, label=nm); left += frac[:, x]
    ax.set_xticks(range(9)); ax.set_xticklabels([f"{zn[c // 3]}\n{pn[c % 3]}" for c in range(9)], fontsize=5.5, rotation=40)
    ax.set_title(f"{NICE[l]}: action mix by situation"); ax.legend(fontsize=6)
if "direct_method_defensive_check" in out:
    pp = out["direct_method_defensive_check"]["per_policy"]; names = list(pp)
    for i, l in enumerate(names):
        m = pp[l]["mean_predicted_danger_steps_vs_cell_average"]; ci = pp[l]["ci95"]
        axs[2].errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=COL[l], capsize=3)
    axs[2].axvline(0, color="k", lw=.8); axs[2].set_yticks(range(len(names))); axs[2].set_yticklabels([NICE[l] for l in names]); axs[2].invert_yaxis()
    axs[2].set_title("Predicted danger steps per interval\nvs situation average (negative = better actions)")
plt.tight_layout(); plt.savefig("figs/secondary_situation_dependence.png", dpi=170)
print("figures saved")

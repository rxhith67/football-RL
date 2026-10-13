"""Tables, confidence intervals and figures for the possession experiments.
usage: bash run.sh analyze_poss.py            (reads results/eval_independent, results/eval_h2h, logs/*.jsonl, checkpoints/*.npz)
Intervals: 95% percentile bootstrap over seeds (independent) or over seed PAIRS (head-to-head). Paired p-values: sign-flip permutation test.
Holm adjustment is applied to the pre-specified family: {possession share, goal difference} x {trained policy vs native}."""
import glob
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M          # grf_rl_coach/metrics.py (read-only import): defensive statistics from the per-step stream

R = Path("results"); FIG = Path("figs"); FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
B = 10000
NICE_CFG = {"A_q_original": "A: Q-learning, original reward (existing)", "B_q_possession": "B: Q-learning, possession reward",
            "C_dq_possession": "C: Double Q, possession reward", "D_dq_original": "D: Double Q, original reward",
            "native": "Native GRF AI", "fixed_compact": "Always compact", "fixed_stretched": "Always stretched",
            "random_switching": "Random switching", "hand_coach": "Hand-written coach"}
SHORT = {"A_q_original": "A  Q/orig", "B_q_possession": "B  Q/poss", "C_dq_possession": "C  DQ/poss", "D_dq_original": "D  DQ/orig",
         "native": "Native", "fixed_compact": "Compact", "fixed_stretched": "Stretched", "random_switching": "Random", "hand_coach": "Hand"}
COL = {"A_q_original": "#7f7f7f", "B_q_possession": "#1b6b45", "C_dq_possession": "#2a9d8f", "D_dq_original": "#8b5cf6", "native": "#c0392b",
       "fixed_compact": "#3b82f6", "fixed_stretched": "#f97316", "random_switching": "#bbbbbb", "hand_coach": "#e0a526"}


def cfg_of(label): return re.sub(r"_run\d+$", "", label)
def nice(label): return NICE_CFG.get(cfg_of(label), label) + (f" [{label.split('_')[-1]}]" if re.search(r"_run\d+$", label) else "")
def color(label): return COL.get(cfg_of(label), "#444")
def order(labels):
    keys = list(NICE_CFG)
    return sorted(labels, key=lambda l: (keys.index(cfg_of(l)) if cfg_of(l) in keys else 99, l))


def boot_ci(x, B=B):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return [float(np.quantile(x[rng.integers(0, len(x), (B, len(x)))].mean(1), q)) for q in (.025, .975)]


def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); d = (a - b)[ok]
    if len(d) < 5: return dict(diff=np.nan, ci=[np.nan, np.nan], p=np.nan, n=int(len(d)))
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    sg = rng.choice([-1.0, 1.0], (B, len(d)))
    p = float(((np.abs((sg * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    return dict(diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], p=p, n=int(len(d)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def match_vals(ep, stream):
    m = M.match_metrics(stream)
    iv = [x for x in ep["interval_shares"] if x is not None]
    return {"possession_share": ep["possession_share"], "goal_diff": ep["goal_diff"], "goals_for": ep["goals_for"], "goals_against": ep["goals_against"],
            "points": ep["points"], "interval_share_mean": float(np.mean(iv)) if iv else np.nan,
            "interval_share_gt50": float(np.mean([x > 0.5 for x in iv])) if iv else np.nan, "match_gt50": float(ep["possession_share"] > 0.5),
            "contested_frac": ep["contested_frac"], "dead_frac": ep["dead_frac"], "return_possession": ep["return_possession"],
            "return_original": ep["return_original"], "recoveries_live": m["recoveries_live"], "time_to_recover_rmst": m["time_to_recover_rmst"],
            "opp_adv_mean": m["opp_adv_mean"], "danger_steps": m["danger_steps"], "shots_conceded": m["shots_conceded"],
            "shots_on_target_conceded": m["shots_on_target_conceded"], "team_length": ep["team_length"]}


def load_eval(d):
    out, seeds = {}, None
    for f in sorted(glob.glob(str(R / d / "*.pkl"))):
        label = Path(f).stem
        data = pickle.load(open(f, "rb"))
        by = {}
        for ep, st in data: by[(ep["seed"], ep["side"])] = (ep, st)
        out[label] = by
    return out


def arr(by, key, side=0):
    seeds = sorted(s for s, sd in by if sd == side)
    return np.array([match_vals(*by[(s, side)])[key] for s in seeds], float), seeds


# =================================================================== independent evaluation
KEYS = ["possession_share", "interval_share_mean", "goal_diff", "goals_for", "goals_against", "points", "recoveries_live", "time_to_recover_rmst",
        "opp_adv_mean", "shots_conceded", "shots_on_target_conceded", "danger_steps"]
out = {}
TAG = os.environ.get("OUT_TAG", "")
EVAL_IND = os.environ.get("EVAL_IND", "eval_independent"); EVAL_H2H = os.environ.get("EVAL_H2H", "eval_h2h")
ind = load_eval(EVAL_IND) if (R / EVAL_IND).exists() else {}
cache = {}
if ind:
    labels = order(list(ind))
    V = {}
    for l in labels:
        seeds = sorted(s for s, sd in ind[l] if sd == 0)
        mv = [match_vals(*ind[l][(s, 0)]) for s in seeds]
        V[l] = {k: np.array([m[k] for m in mv], float) for k in mv[0]}
        V[l]["_ep"] = [ind[l][(s, 0)][0] for s in seeds]
    n = len(next(iter(V.values()))["goal_diff"])
    out["independent"] = {"n_matches_per_policy": n, "seeds": [seeds[0], seeds[-1]], "summary": {}}
    for l in labels:
        v = V[l]; gd = v["goal_diff"]
        ac = np.sum([e["action_counts"] for e in v["_ep"]], 0)
        s = {"possession_share": [float(v["possession_share"].mean()), float(v["possession_share"].std(ddof=1) / np.sqrt(n)), boot_ci(v["possession_share"])],
             "matches_over_50pct_possession": [int(v["match_gt50"].sum()), n, wilson(int(v["match_gt50"].sum()), n)],
             "interval_possession_mean": float(np.nanmean(v["interval_share_mean"])), "intervals_over_50pct": float(np.nanmean(v["interval_share_gt50"])),
             "goal_diff": [float(gd.mean()), float(gd.std(ddof=1) / np.sqrt(n)), boot_ci(gd)],
             "W/D/L": [int((gd > 0).sum()), int((gd == 0).sum()), int((gd < 0).sum())],
             "goals_for": float(v["goals_for"].mean()), "goals_against": float(v["goals_against"].mean()), "points_per_match": float(v["points"].mean()),
             "contested_frac": float(v["contested_frac"].mean()), "dead_frac": float(v["dead_frac"].mean()),
             "return_possession_reward": float(v["return_possession"].mean()), "return_original_reward": float(v["return_original"].mean()),
             "recoveries_live": float(v["recoveries_live"].mean()), "time_to_recover_rmst": float(np.nanmean(v["time_to_recover_rmst"])),
             "opp_adv_mean": float(np.nanmean(v["opp_adv_mean"])), "shots_conceded": float(v["shots_conceded"].mean()),
             "danger_steps": float(v["danger_steps"].mean()), "team_length": float(v["team_length"].mean()),
             "action_fraction_compact_default_stretched": (ac / ac.sum()).round(3).tolist()}
        out["independent"]["summary"][l] = s
    # configuration-level averages across independent training runs (per-seed mean of the runs' values)
    groups = {}
    for l in labels: groups.setdefault(cfg_of(l), []).append(l)
    G = {c: {k: np.mean([V[l][k] for l in ls], 0) for k in KEYS} for c, ls in groups.items()}
    out["independent"]["config_runs"] = {c: ls for c, ls in groups.items()}
    comps = {}
    def add(name, a, b, source=G):
        if a in source and b in source:
            comps[name] = {k: paired(source[a][k], source[b][k]) for k in KEYS}
    for c in groups:
        if c not in ("native",): add(f"{c} minus native", c, "native")
    for c in ("B_q_possession", "C_dq_possession", "D_dq_original"):
        add(f"{c} minus A_q_original", c, "A_q_original")
    add("C_dq_possession minus B_q_possession  (RQ2: Double Q vs Q, possession reward)", "C_dq_possession", "B_q_possession")
    add("B_q_possession minus A_q_original  (RQ1: possession reward, Q-learning)", "B_q_possession", "A_q_original")
    add("C_dq_possession minus D_dq_original  (reward effect within Double Q)", "C_dq_possession", "D_dq_original")
    add("D_dq_original minus A_q_original  (algorithm effect, original reward)", "D_dq_original", "A_q_original")
    for c in ("B_q_possession", "C_dq_possession", "D_dq_original", "A_q_original"):
        for f in ("fixed_compact", "fixed_stretched"): add(f"{c} minus {f}", c, f)
    # per-run comparisons (to expose run-to-run variability)
    for l in labels:
        if re.search(r"_run\d+$", l): comps[f"{l} minus native"] = {k: paired(V[l][k], V["native"][k]) for k in KEYS} if "native" in V else {}
    prim = [(c, k) for c in ("B_q_possession", "C_dq_possession", "D_dq_original") for k in ("possession_share", "goal_diff") if f"{c} minus native" in comps]
    if prim:
        adj = holm([comps[f"{c} minus native"][k]["p"] for c, k in prim])
        for (c, k), a in zip(prim, adj): comps[f"{c} minus native"][k]["p_holm"] = float(a)
    out["independent"]["comparisons"] = comps
    # observation dependence is NOT part of this experiment's spec; action frequencies are reported in the summary

# =================================================================== head-to-head
h2h = load_eval(EVAL_H2H) if (R / EVAL_H2H).exists() else {}
if h2h:
    labels_h = order(list(h2h))
    H = {}
    for l in labels_h:
        seeds = sorted({s for s, sd in h2h[l]})
        mv = {sd: [match_vals(*h2h[l][(s, sd)]) for s in seeds] for sd in (0, 1)}
        H[l] = {"seeds": seeds, **{f"{sd}_{k}": np.array([m[k] for m in mv[sd]], float) for sd in (0, 1) for k in ("goal_diff", "possession_share", "goals_for", "goals_against", "points", "contested_frac", "return_possession", "return_original")}}
        H[l]["_ep"] = {sd: [h2h[l][(s, sd)][0] for s in seeds] for sd in (0, 1)}
    n2 = len(H[labels_h[0]]["seeds"])
    out["headtohead"] = {"n_seed_pairs": n2, "seeds": [H[labels_h[0]]["seeds"][0], H[labels_h[0]]["seeds"][-1]], "summary": {}}
    for l in labels_h:
        h = H[l]; s = {}
        for name, sel in (("left", [0]), ("right", [1]), ("both", [0, 1])):
            gd = np.concatenate([h[f"{sd}_goal_diff"] for sd in sel]); ps = np.concatenate([h[f"{sd}_possession_share"] for sd in sel])
            gf = np.concatenate([h[f"{sd}_goals_for"] for sd in sel]); ga = np.concatenate([h[f"{sd}_goals_against"] for sd in sel])
            if name == "both":   # resample seed pairs so the two legs of a seed stay together
                pm_gd = (h["0_goal_diff"] + h["1_goal_diff"]) / 2; pm_ps = (h["0_possession_share"] + h["1_possession_share"]) / 2
                gd_ci, ps_ci = boot_ci(pm_gd), boot_ci(pm_ps)
            else: gd_ci, ps_ci = boot_ci(gd), boot_ci(ps)
            s[name] = {"matches": int(len(gd)), "W/D/L": [int((gd > 0).sum()), int((gd == 0).sum()), int((gd < 0).sum())],
                       "W/D/L_pct": [float(100 * (gd > 0).mean()), float(100 * (gd == 0).mean()), float(100 * (gd < 0).mean())],
                       "rl_goals": float(gf.mean()), "native_goals": float(ga.mean()), "goal_diff": [float(gd.mean()), gd_ci],
                       "rl_possession": [float(ps.mean()), ps_ci], "native_possession": float(1 - ps.mean()),
                       "matches_over_50pct_possession": [int((ps > 0.5).sum()), int(len(ps)), float(100 * (ps > 0.5).mean())]}
        pg = h["0_goal_diff"] + h["1_goal_diff"]              # seed-pair outcome: combined goal difference over the two legs
        s["pair_outcomes_by_combined_goal_diff_win/draw/loss"] = [int((pg > 0).sum()), int((pg == 0).sum()), int((pg < 0).sum())]
        c = h["0_goal_diff"] - h["1_goal_diff"]
        s["side_contrast_goal_diff_left_minus_right"] = [float(c.mean()), boot_ci(c)]
        s["side_contrast_possession_left_minus_right"] = [float((h["0_possession_share"] - h["1_possession_share"]).mean()), boot_ci(h["0_possession_share"] - h["1_possession_share"])]
        ac = {sd: np.sum([e["action_counts"] for e in h["_ep"][sd]], 0) for sd in (0, 1)}
        s["action_fraction_left"] = (ac[0] / ac[0].sum()).round(3).tolist(); s["action_fraction_right"] = (ac[1] / ac[1].sum()).round(3).tolist()
        out["headtohead"]["summary"][l] = s
    PM = {l: {"goal_diff": (H[l]["0_goal_diff"] + H[l]["1_goal_diff"]) / 2, "possession_share": (H[l]["0_possession_share"] + H[l]["1_possession_share"]) / 2} for l in labels_h}
    gh = {}
    for l in labels_h: gh.setdefault(cfg_of(l), []).append(l)
    GH = {c: {k: np.mean([PM[l][k] for l in ls], 0) for k in ("goal_diff", "possession_share")} for c, ls in gh.items()}
    comps_h = {}
    def addh(name, a, b):
        if a in GH and b in GH: comps_h[name] = {k: paired(GH[a][k], GH[b][k]) for k in ("goal_diff", "possession_share")}
    for c in gh:
        if c != "native": addh(f"{c} minus native (pair mean)", c, "native")
    for c in ("B_q_possession", "C_dq_possession", "D_dq_original"): addh(f"{c} minus A_q_original (pair mean)", c, "A_q_original")
    addh("C_dq_possession minus B_q_possession (pair mean)", "C_dq_possession", "B_q_possession")
    addh("C_dq_possession minus D_dq_original (pair mean)", "C_dq_possession", "D_dq_original")
    out["headtohead"]["comparisons"] = comps_h
    out["headtohead"]["config_runs"] = gh

json.dump(out, open(R / f"possession_analysis{TAG}.json", "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))

# =================================================================== tables
L = []
if "independent" in out:
    I = out["independent"]; labs = list(I["summary"])
    L += [f"## Independent evaluation: each policy plays the native AI on the left; {I['n_matches_per_policy']} matches per policy, seeds {I['seeds'][0]} to {I['seeds'][1]} (identical for all policies)", "",
          "| Policy | Possession % (mean [95% CI]) | Matches >50% possession | Goal diff (mean [95% CI]) | W-D-L | Goals for / against | Points | Actions C/D/S |", "|---|---|---|---|---|---|---|---|"]
    for l in labs:
        s = I["summary"][l]; a = s["action_fraction_compact_default_stretched"]
        L.append(f"| {nice(l)} | {100 * s['possession_share'][0]:.1f} [{100 * s['possession_share'][2][0]:.1f}, {100 * s['possession_share'][2][1]:.1f}] | "
                 f"{s['matches_over_50pct_possession'][0]}/{s['matches_over_50pct_possession'][1]} ({100 * s['matches_over_50pct_possession'][0] / s['matches_over_50pct_possession'][1]:.0f}%) | "
                 f"{s['goal_diff'][0]:+.3f} [{s['goal_diff'][2][0]:+.2f}, {s['goal_diff'][2][1]:+.2f}] | {'-'.join(map(str, s['W/D/L']))} | {s['goals_for']:.2f} / {s['goals_against']:.2f} | {s['points_per_match']:.2f} | {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} |")
    L += ["", "Other measured outcomes (means per match; contested = nobody owns the ball, dead = kick-off/throw-in etc.):", "",
          "| Policy | Interval possession % | Contested % | Dead-ball % | Possession-reward return | Original-reward return | Live recoveries | Time to recover (RMST) | Opp. advancement | Shots conceded (heuristic) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for l in labs:
        s = I["summary"][l]
        L.append(f"| {nice(l)} | {100 * s['interval_possession_mean']:.1f} | {100 * s['contested_frac']:.1f} | {100 * s['dead_frac']:.1f} | {s['return_possession_reward']:+.2f} | {s['return_original_reward']:+.2f} | {s['recoveries_live']:.2f} | {s['time_to_recover_rmst']:.1f} | {s['opp_adv_mean']:.3f} | {s['shots_conceded']:.2f} |")
    L += ["", "Paired differences (same seeds; A minus B; mean [95% CI], sign-flip p; Holm-adjusted p for the pre-specified family where shown):", ""]
    for name, cc in I["comparisons"].items():
        if re.search(r"_run\d+ minus native", name): continue
        P_, G_ = cc["possession_share"], cc["goal_diff"]
        extra = lambda c: f", Holm p {c['p_holm']:.3f}" if "p_holm" in c else ""
        L.append(f"- **{name}**: possession {100 * P_['diff']:+.2f} pts [{100 * P_['ci'][0]:+.2f}, {100 * P_['ci'][1]:+.2f}] (p = {P_['p']:.3f}{extra(P_)}); "
                 f"goal diff {G_['diff']:+.3f} [{G_['ci'][0]:+.2f}, {G_['ci'][1]:+.2f}] (p = {G_['p']:.3f}{extra(G_)}); goals against {cc['goals_against']['diff']:+.3f} [{cc['goals_against']['ci'][0]:+.2f}, {cc['goals_against']['ci'][1]:+.2f}]")
    runs = {c: ls for c, ls in I["config_runs"].items() if len(ls) > 1}
    if runs:
        L += ["", "Run-to-run variability (each independent training run against native):", ""]
        for c, ls in runs.items():
            for l in ls:
                cc = I["comparisons"].get(f"{l} minus native", {})
                if cc: L.append(f"- {nice(l)}: possession {100 * cc['possession_share']['diff']:+.2f} pts [{100 * cc['possession_share']['ci'][0]:+.2f}, {100 * cc['possession_share']['ci'][1]:+.2f}]; goal diff {cc['goal_diff']['diff']:+.3f} [{cc['goal_diff']['ci'][0]:+.2f}, {cc['goal_diff']['ci'][1]:+.2f}]")
if "headtohead" in out:
    Hh = out["headtohead"]; labs = list(Hh["summary"])
    L += ["", f"## Head-to-head: RL team vs native AI on both sides, {Hh['n_seed_pairs']} seed pairs ({2 * Hh['n_seed_pairs']} matches per policy), seeds {Hh['seeds'][0]} to {Hh['seeds'][1]}", "",
          "| RL policy | Side | Matches | W-D-L | RL goals | Native goals | RL goal diff [95% CI] | RL possession % [95% CI] | Native possession % | Matches >50% possession |", "|---|---|---|---|---|---|---|---|---|---|"]
    for l in labs:
        for sd in ("left", "right", "both"):
            s = Hh["summary"][l][sd]
            L.append(f"| {nice(l)} | {sd} | {s['matches']} | {'-'.join(map(str, s['W/D/L']))} ({s['W/D/L_pct'][0]:.0f}/{s['W/D/L_pct'][1]:.0f}/{s['W/D/L_pct'][2]:.0f}%) | {s['rl_goals']:.3f} | {s['native_goals']:.3f} | {s['goal_diff'][0]:+.3f} [{s['goal_diff'][1][0]:+.2f}, {s['goal_diff'][1][1]:+.2f}] | "
                     f"{100 * s['rl_possession'][0]:.1f} [{100 * s['rl_possession'][1][0]:.1f}, {100 * s['rl_possession'][1][1]:.1f}] | {100 * s['native_possession']:.1f} | {s['matches_over_50pct_possession'][0]}/{s['matches_over_50pct_possession'][1]} ({s['matches_over_50pct_possession'][2]:.0f}%) |")
    L += ["", "Tactical action frequencies of each RL policy (compact / native default / stretched):", ""]
    for l in labs:
        s_ = Hh["summary"][l]; fl, fr_ = s_["action_fraction_left"], s_["action_fraction_right"]
        L.append(f"- {nice(l)}: left {fl[0]:.2f}/{fl[1]:.2f}/{fl[2]:.2f}; right {fr_[0]:.2f}/{fr_[1]:.2f}/{fr_[2]:.2f}")
    L += ["", "Seed-pair outcomes (combined goal difference of the two legs; win/draw/loss for the RL policy):", ""]
    for l in labs:
        po = Hh["summary"][l]["pair_outcomes_by_combined_goal_diff_win/draw/loss"]
        L.append(f"- {nice(l)}: {po[0]}-{po[1]}-{po[2]} of {Hh['n_seed_pairs']} seed pairs")
    L += ["", "Side contrast (left leg minus right leg, per seed pair). For `native` both legs are the same match seen from either side, so this equals twice the intrinsic left-side advantage of the scenario (right-side players have difficulty 0.95 versus 1.0 on the left):", ""]
    for l in labs:
        c = Hh["summary"][l]["side_contrast_goal_diff_left_minus_right"]; cp = Hh["summary"][l]["side_contrast_possession_left_minus_right"]
        L.append(f"- {nice(l)}: goal diff {c[0]:+.3f} [{c[1][0]:+.2f}, {c[1][1]:+.2f}]; possession {100 * cp[0]:+.2f} pts [{100 * cp[1][0]:+.2f}, {100 * cp[1][1]:+.2f}]")
    L += ["", "Pair-mean differences between RL policies and native / original Q (paired over seed pairs):", ""]
    for name, cc in Hh["comparisons"].items():
        L.append(f"- **{name}**: goal diff {cc['goal_diff']['diff']:+.3f} [{cc['goal_diff']['ci'][0]:+.2f}, {cc['goal_diff']['ci'][1]:+.2f}] (p = {cc['goal_diff']['p']:.3f}); possession {100 * cc['possession_share']['diff']:+.2f} pts [{100 * cc['possession_share']['ci'][0]:+.2f}, {100 * cc['possession_share']['ci'][1]:+.2f}] (p = {cc['possession_share']['p']:.3f})")
Path(R / f"possession_tables{TAG}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

# =================================================================== training curves
logs = {}
for f in sorted(glob.glob("logs/*_run[0-9]*.jsonl")):
    tag = Path(f).stem
    if tag.startswith("smoke") or "run99" in tag: continue
    logs[tag] = [json.loads(l) for l in open(f)]
CFGTAG = {"q_possession": "B_q_possession", "double_q_possession": "C_dq_possession", "double_q_original": "D_dq_original"}
if logs:
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    k = 10
    for tag, L_ in logs.items():
        cfg = CFGTAG.get(re.sub(r"_run\d+$", "", tag), tag); ep = np.array([r["episodes"] for r in L_])
        ls = "-" if tag.endswith("run1") else "--"
        for a, key, ttl in ((ax[0], "train_possession_share", "Training possession share (matches under exploration)"), (ax[1], "train_goal_diff", "Training goal difference per match"),
                            (ax[2], "train_return_trained_on", "Return in the reward each agent trains on")):
            v = np.array([r[key] for r in L_]); kk = min(k, len(v)); sm = np.convolve(v, np.ones(kk) / kk, mode="valid")
            a.plot(ep[kk - 1:], sm, ls, color=color(cfg), label=f"{SHORT.get(cfg, cfg)} {tag[-4:]}"); a.set_title(ttl, fontsize=8.5); a.set_xlabel("training matches")
        dv = [(r["episodes"], r["dev_possession_share"]) for r in L_ if "dev_possession_share" in r]
        ax[3].plot(*zip(*dv), "o" + ls, color=color(cfg), label=f"{SHORT.get(cfg, cfg)} {tag[-4:]}")
    ax[3].set_title("Greedy policy possession share on 6 dev seeds", fontsize=8.5); ax[3].set_xlabel("training matches"); ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(FIG / f"poss_training_curves{TAG}.png", dpi=160); plt.close()
    tq = {}
    for tag in logs:
        z = np.load(f"checkpoints/{tag}.npz"); q = z["q"]; vis = z["visits"].sum(1)
        g = np.array([1 if np.isclose(q[s], q[s].max()).sum() > 1 and np.isclose(q[s][1], q[s].max()) else int(np.argmax(q[s])) for s in range(243)])
        tq[tag] = {"iter": int(z["iter"]), "episodes": int(z["episodes"]), "decisions": int(z["decisions"]), "engine_steps": int(z["engine_steps"]), "wall_hours": float(z["wall"]) / 3600,
                   "greedy_counts": np.bincount(g, minlength=3).tolist(), "visit_weighted_greedy": (np.bincount(g, weights=vis, minlength=3) / vis.sum()).round(3).tolist(),
                   "states_visited": int((vis > 0).sum()), "median_visits_per_state_action": float(np.median(z["visits"][z["visits"] > 0]))}
        tq[tag]["_g"] = g
    tags = list(tq)
    agree = {f"{a} vs {b}": float((tq[a]["_g"] == tq[b]["_g"]).mean()) for i, a in enumerate(tags) for b in tags[i + 1:]}
    a0 = np.load("../grf_rl_coach/checkpoints/q_it100_final.npz")["q"]
    gA = np.array([1 if np.isclose(a0[s], a0[s].max()).sum() > 1 and np.isclose(a0[s][1], a0[s].max()) else int(np.argmax(a0[s])) for s in range(243)])
    for t in tags: agree[f"{t} vs A_q_original"] = float((tq[t]["_g"] == gA).mean())
    for t in tags: tq[t].pop("_g")
    json.dump({"training": tq, "greedy_policy_agreement_fraction_of_243_states": agree}, open(R / f"training_summary{TAG}.json", "w"), indent=1)
    print(json.dumps({"training": tq, "agreement": agree}, indent=1))

# =================================================================== figures
if "independent" in out:
    I = out["independent"]; labs = [l for l in I["summary"]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, l in enumerate(labs):
        s = I["summary"][l]
        ax[0].errorbar(100 * s["possession_share"][0], i, xerr=[[100 * (s["possession_share"][0] - s["possession_share"][2][0])], [100 * (s["possession_share"][2][1] - s["possession_share"][0])]], fmt="o", color=color(l), capsize=3)
        ax[1].errorbar(s["goal_diff"][0], i, xerr=[[s["goal_diff"][0] - s["goal_diff"][2][0]], [s["goal_diff"][2][1] - s["goal_diff"][0]]], fmt="o", color=color(l), capsize=3)
        ax[2].errorbar(100 * s["possession_share"][0], s["goal_diff"][0], xerr=[[100 * (s["possession_share"][0] - s["possession_share"][2][0])], [100 * (s["possession_share"][2][1] - s["possession_share"][0])]],
                       yerr=[[s["goal_diff"][0] - s["goal_diff"][2][0]], [s["goal_diff"][2][1] - s["goal_diff"][0]]], fmt="o", color=color(l), capsize=2, label=SHORT.get(cfg_of(l), l) + (l[-5:] if "_run" in l else ""))
    for a in ax[:2]:
        a.set_yticks(range(len(labs))); a.set_yticklabels([nice(l) for l in labs] if a is ax[0] else [], fontsize=7); a.invert_yaxis()
    ax[0].axvline(50, color="k", lw=.8, ls="--"); ax[0].set_xlabel("our share of controlled possession (%), mean with 95% CI"); ax[0].set_title("Possession")
    ax[1].axvline(0, color="k", lw=.8); ax[1].set_xlabel("goal difference per match, mean with 95% CI"); ax[1].set_title("Goal difference")
    ax[2].axhline(0, color="k", lw=.5); ax[2].axvline(50, color="k", lw=.5, ls="--"); ax[2].set_xlabel("possession %"); ax[2].set_ylabel("goal difference"); ax[2].set_title("Possession vs goal difference"); ax[2].legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / f"poss_independent_summary{TAG}.png", dpi=170); plt.close()
    fig, ax = plt.subplots(figsize=(9, 4)); A = np.array([I["summary"][l]["action_fraction_compact_default_stretched"] for l in labs]); left = np.zeros(len(labs))
    for j, (nme, c) in enumerate((("compact", "#3b82f6"), ("native default", "#cbd5e1"), ("stretched", "#f97316"))):
        ax.barh(range(len(labs)), A[:, j], left=left, color=c, label=nme); left += A[:, j]
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([nice(l) for l in labs], fontsize=7); ax.invert_yaxis(); ax.legend(fontsize=7, ncol=3); ax.set_title("Tactical decision frequencies")
    plt.tight_layout(); plt.savefig(FIG / f"poss_action_frequencies{TAG}.png", dpi=160); plt.close()
if "headtohead" in out:
    Hh = out["headtohead"]; labs = list(Hh["summary"]); fig, ax = plt.subplots(1, 2, figsize=(14, 4.3)); w = 0.26
    for j, (sd, c) in enumerate((("left", "#3b82f6"), ("right", "#f97316"), ("both", "#1b6b45"))):
        for key, a, scale in (("goal_diff", ax[0], 1.0), ("rl_possession", ax[1], 100.0)):
            vals = []; lo = []; hi = []
            for l in labs:
                s = Hh["summary"][l][sd]
                m, ci = (s["goal_diff"][0], s["goal_diff"][1]) if key == "goal_diff" else (s["rl_possession"][0] * scale, [x * scale for x in s["rl_possession"][1]])
                vals.append(m); lo.append(m - ci[0]); hi.append(ci[1] - m)
            a.bar(np.arange(len(labs)) + (j - 1) * w, vals, w, yerr=[lo, hi], color=c, capsize=2, label=sd)
    for a, t in ((ax[0], "RL goal difference vs native (per match)"), (ax[1], "RL possession % of controlled time")):
        a.set_xticks(range(len(labs))); a.set_xticklabels([SHORT.get(cfg_of(l), l) + (l[-5:] if "_run" in l else "") for l in labs], fontsize=7, rotation=20); a.set_title(t); a.legend(fontsize=7)
    ax[0].axhline(0, color="k", lw=.8); ax[1].axhline(50, color="k", lw=.8, ls="--"); ax[1].set_ylim(40, 60)
    plt.tight_layout(); plt.savefig(FIG / f"poss_headtohead{TAG}.png", dpi=170); plt.close()
print("figures saved")

"""Football-statistics analysis of the recorded traces (no engine). Two modes with different rules:

  python stats_analyze.py --set dev     DEVELOPMENT diagnostics only (seeds 1,100,000+): measurement validity, noise, proxy recall,
                                        state-dependence audit. Used to FREEZE eval_spec.json. Not evidence about the policy.
  python stats_analyze.py --set final   CONFIRMATORY analysis (seeds 1,000,000+). Reads eval_spec.json and refuses to run if its
                                        sha256 differs from the one recorded when the final matches were collected. Metrics the spec
                                        did not name as primary are labelled secondary/exploratory whatever their results look like.
Comparisons are paired by seed (identical match seeds for every policy). Both paired and unpaired bootstrap CIs are reported because
different actions make the matches diverge, so pairing buys little variance reduction (see 'seed correlation')."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stats_metrics as sm

R = Path("results"); FIGS = Path("figs")
LABELS = ("native_unmodified", "fixed_compact", "fixed_stretched", "hand_coach", "random_switching", "q_learning", "q_learning_shuffled_obs")
NICE = {"native_unmodified": "Native GRF AI", "fixed_compact": "Fixed compact", "fixed_stretched": "Fixed stretched", "hand_coach": "Hand-written adaptive",
        "random_switching": "Random switching", "q_learning": "Q-learning (trained)", "q_learning_shuffled_obs": "Q-learning, shuffled obs."}
COL = {"native_unmodified": "#6b7280", "fixed_compact": "#3b82f6", "fixed_stretched": "#f97316", "hand_coach": "#a855f7",
       "random_switching": "#9ca3af", "q_learning": "#15803d", "q_learning_shuffled_obs": "#86efac"}
# key: (display name, better direction (+1 higher is better, -1 lower is better, 0 descriptive), unit/format)
METRICS = {
    "goals_against": ("Goals conceded", -1, "{:.2f}"), "goals_for": ("Goals scored", +1, "{:.2f}"), "goal_diff": ("Goal difference", +1, "{:+.2f}"),
    "poss_carry_share": ("Possession % (last-owner carry-forward)", +1, "{:.3f}"), "poss_owned_share": ("Possession % (owned steps only)", +1, "{:.3f}"),
    "frac_own": ("Steps ball owned by us", 0, "{:.3f}"), "frac_opp": ("Steps ball owned by opponent", 0, "{:.3f}"), "frac_loose": ("Steps ball loose/in flight", 0, "{:.3f}"),
    "recoveries_open": ("Possession recoveries, open play", +1, "{:.1f}"), "recoveries_all": ("Possession recoveries, incl. restarts", +1, "{:.1f}"),
    "recoveries_direct": ("Recoveries, direct owner flip (legacy def.)", +1, "{:.1f}"), "losses_all": ("Possession losses (all)", -1, "{:.1f}"),
    "t_recover_mean": ("Mean steps to regain possession", -1, "{:.1f}"), "t_recover_open_mean": ("Mean steps to regain, open-play periods", -1, "{:.1f}"),
    "opp_def_third_steps": ("Opp. possession steps in our defensive third", -1, "{:.0f}"), "opp_def_third_entries": ("Opp. spells reaching our defensive third", -1, "{:.1f}"),
    "opp_box_entries": ("Opp. spells entering our penalty area", -1, "{:.2f}"), "opp_box_steps": ("Opp. possession steps in our penalty area", -1, "{:.0f}"),
    "opp_spell_min_x_mean": ("Opp. deepest ball x per spell (mean)", +1, "{:+.3f}"), "ball_x_mean": ("Mean ball x, normal play (territory)", +1, "{:+.3f}"),
    "opp_shot_attempts": ("Opp. shot-like attempts (proxy)", -1, "{:.2f}"), "opp_shots_on_target": ("Opp. shot-like on target (proxy)", -1, "{:.2f}"),
    "own_shot_attempts": ("Our shot-like attempts (proxy)", +1, "{:.2f}"), "own_shots_on_target": ("Our shot-like on target (proxy)", +1, "{:.2f}"),
}
B = 10000
RNG = np.random.default_rng(2024)


# ------------------------------------------------------------------ loading / per-match metrics
def load(setname):
    d = R / f"stats_{setname}"
    data = {}
    for l in LABELS:
        f = d / f"{l}.pkl"
        if not f.exists():
            continue
        z = pickle.load(open(f, "rb"))
        rows, evs = [], []
        for o, tr in zip(z["outs"], z["traces"]):
            m, ev = sm.compute_match(tr)
            assert (m["goals_for"], m["goals_against"]) == (o["goals_for"], o["goals_against"]), "trace score != env score"
            m["seed"] = o["seed"]; rows.append(m); evs.append(ev)
        data[l] = {"rows": rows, "events": evs, "traces": z["traces"], "outs": z["outs"], "seeds": z["seeds"], "meta": z["meta"]}
    return data


def col(data, l, k):
    return np.array([r[k] for r in data[l]["rows"]], float)


# ------------------------------------------------------------------ statistics
def boot_ci(x, stat=np.nanmean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = np.array([stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(2000)]) if stat is not np.nanmean else x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def compare(a, b):
    """a, b: per-seed arrays (same seed order). Paired bootstrap CI + sign-flip p, unpaired bootstrap CI, dz, seed correlation."""
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]; n = len(a)
    d = a - b
    bs = d[RNG.integers(0, n, (B, n))].mean(1)
    flips = (RNG.integers(0, 2, (B, n)) * 2 - 1) * d
    p = float((np.abs(flips.mean(1)) >= abs(d.mean()) - 1e-12).mean())
    ub = a[RNG.integers(0, n, (B, n))].mean(1) - b[RNG.integers(0, n, (B, n))].mean(1)
    sd = d.std(ddof=1)
    return {"diff": float(d.mean()), "paired_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "unpaired_ci95": [float(np.quantile(ub, .025)), float(np.quantile(ub, .975))], "p_signflip": p,
            "dz": float(d.mean() / sd) if sd > 0 else float("nan"), "seed_corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"), "n": int(n)}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


def km(durations, censored):
    """Kaplan-Meier survival of 'still without the ball'. Returns step times and S(t)."""
    d = np.asarray(durations, float); c = np.asarray(censored, bool)
    ts = np.unique(d[~c]); s = 1.0; xs, ys = [0.0], [1.0]
    for t in ts:
        at_risk = (d >= t).sum(); ev = ((d == t) & ~c).sum()
        s *= 1 - ev / at_risk; xs.append(t); ys.append(s)
    return np.array(xs), np.array(ys)


def pooled_recovery(data, l, idx, open_only=False):
    """Mean completed time-to-regain (steps) pooling all periods of the matches in idx (cluster bootstrap unit = match)."""
    vals = []
    for i in idx:
        vals += [dur for _, dur, cens, op in data[l]["events"][i]["oop"] if not cens and (op or not open_only)]
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------ state-dependence
def discretise_feats(F):
    """(zone, possession, score, phase) as in coach_env.discretize but WITHOUT the last-action component."""
    zone = np.where(F[:, 0] < -0.33, 0, np.where(F[:, 0] <= 0.33, 1, 2)); poss = np.argmax(F[:, 2:5], 1)
    sd = F[:, 5] * 3.0; score = np.where(sd < -0.5, 0, np.where(sd < 0.5, 1, 2))
    t = F[:, 6]; phase = np.where(t > 2 / 3, 0, np.where(t > 1 / 3, 1, 2))
    return zone, poss, score, phase


def nmi(a, s, nperm=500):
    """I(A;S)/H(A) with a permutation-null correction: returns (observed, null mean, excess, p)."""
    def mi(a, s):
        n = len(a); ja = np.unique(a, return_inverse=True)[1]; js = np.unique(s, return_inverse=True)[1]
        tab = np.zeros((ja.max() + 1, js.max() + 1)); np.add.at(tab, (ja, js), 1); p = tab / n
        pa, ps = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
        h = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
        return (p[nz] * np.log(p[nz] / (pa @ ps)[nz])).sum() / h if h > 1e-9 else 0.0     # constant policy: undefined -> 0
    obs = mi(a, s); null = np.array([mi(RNG.permutation(a), s) for _ in range(nperm)])
    return float(obs), float(null.mean()), float(obs - null.mean()), float((null >= obs).mean())


def state_dependence(data):
    out = {}
    for l, d in data.items():
        A = np.concatenate([t["dec_action"] for t in d["traces"]]).astype(int); F = np.concatenate([t["feats"] for t in d["traces"]])
        z, p, sc, ph = discretise_feats(F); S = ((z * 3 + p) * 3 + sc) * 3 + ph
        obs, null, exc, pv = nmi(A, S)
        frac = np.bincount(A, minlength=3) / len(A)
        out[l] = {"n_decisions": int(len(A)), "action_frac_compact_default_stretched": frac.round(3).tolist(), "nmi": obs, "nmi_null": null, "nmi_excess": exc, "p_perm": pv,
                  "by_zone_poss": [[(np.bincount(A[(z == zz) & (p == pp)], minlength=3) / max(1, ((z == zz) & (p == pp)).sum())).round(3).tolist() for pp in range(3)] for zz in range(3)]}
    return out


def q_audit(data, ckpt="checkpoints/q_it100_final.npz"):
    z = np.load(ckpt); q, vis = z["q"], z["visits"]
    d = data["q_learning"]
    S = np.concatenate([t["dec_state"] for t in d["traces"]]).astype(int); A = np.concatenate([t["dec_action"] for t in d["traces"]]).astype(int)
    tied = np.array([np.isclose(q[s], q[s].max()).all() for s in S])
    greedy = np.array([int(np.argmax(q[s])) for s in S])
    srt = np.sort(q[S], 1); margin = srt[:, -1] - srt[:, -2]
    visited_states = np.flatnonzero(vis.sum(1) > 0)
    return {"training_decisions_total": int(vis.sum()), "states_visited_in_training": int(len(visited_states)), "n_states": int(q.shape[0]),
            "eval_decisions": int(len(S)), "eval_distinct_states": int(len(np.unique(S))),
            "frac_eval_decisions_in_all_tied_state": float(tied.mean()), "frac_chosen_equals_native_default": float((A == 1).mean()),
            "chosen_action_training_visits_median": float(np.median(vis[S, A])), "frac_chosen_action_visits_lt_30": float((vis[S, A] < 30).mean()),
            "greedy_margin_median": float(np.median(margin)), "greedy_margin_p90": float(np.quantile(margin, .9)),
            "frac_visited_states_greedy_nondefault": float(np.mean([int(np.argmax(q[s])) != 1 and not np.isclose(q[s], q[s].max()).all() for s in visited_states])),
            "q_value_spread_abs_max": float(np.abs(q).max())}


# ------------------------------------------------------------------ dev diagnostics
def dev_diagnostics(data):
    out = {"n_matches_per_policy": len(data["native_unmodified"]["rows"]), "metrics": {}, "shot_proxy": {}}
    for k, (nm, better, fmt) in METRICS.items():
        per = {l: col(data, l, k) for l in data}
        allv = np.concatenate([v[~np.isnan(v)] for v in per.values()])
        within = np.mean([np.nanvar(v, ddof=1) for v in per.values()])
        out["metrics"][k] = {"name": nm, "mean_by_policy": {l: float(np.nanmean(v)) for l, v in per.items()}, "sd_within_policy": float(np.sqrt(within)),
                             "cv_within": float(np.sqrt(within) / abs(np.nanmean(allv))) if np.nanmean(allv) != 0 else None,
                             "n_nan": int(sum(np.isnan(v).sum() for v in per.values())),
                             "paired_sd_q_minus_native": float(np.nanstd(per["q_learning"] - per["native_unmodified"], ddof=1)) if "q_learning" in per else None}
        out["metrics"][k]["mde_80pct_n200"] = 2.8 * out["metrics"][k]["paired_sd_q_minus_native"] / np.sqrt(200) if out["metrics"][k]["paired_sd_q_minus_native"] else None
    H = T = 0; shots = []
    for l in data:
        for ev in data[l]["events"]:
            h, t = sm.goal_recall(ev); H += h; T += t; shots += ev["shots_opp"] + ev["shots_own"]
    out["shot_proxy"] = {"goals_total": T, "goals_preceded_by_on_target_event_25steps": H, "recall": H / max(1, T), "n_events": len(shots),
                         "on_target_frac": float(np.mean([e["on_target"] for e in shots])) if shots else None,
                         "outcome_counts": {o: int(sum(e["outcome"] == o for e in shots)) for o in ("saved_or_blocked", "out")},
                         "on_target_frac_no_defender_touch_15steps": float(np.mean([e["outcome"] == "out" for e in shots if e["on_target"]])) if shots else None,
                         "on_target_followed_by_goal_25steps": float(np.mean([any(0 < t - e["t"] <= 25 and k == e["team"] for t, k in ev["goals"]) for l in data for ev in data[l]["events"] for e in ev["shots_opp"] + ev["shots_own"] if e["on_target"]]))}
    # spell-length sanity for the min_spell choice
    runs = {0: [], 1: []}
    for l in data:
        for tr in data[l]["traces"]:
            for k, s, e in sm._runs(tr["owner"]):
                if k in runs: runs[k].append(e - s + 1)
    out["owned_run_length"] = {str(k): {"n": len(v), "frac_len1": float(np.mean(np.array(v) == 1)), "median": float(np.median(v)), "p90": float(np.quantile(v, .9))} for k, v in runs.items()}
    # sensitivity of counts to min_spell
    sens = {}
    for ms in (1, 2, 3, 5):
        p = dict(sm.P, min_spell=ms); vals = [sm.compute_match(tr, p)[0]["recoveries_open"] for tr in data["native_unmodified"]["traces"]]
        sens[ms] = float(np.mean(vals))
    out["recoveries_open_native_by_min_spell"] = sens
    out["state_dependence"] = state_dependence(data)
    out["q_audit"] = q_audit(data)
    return out


# ------------------------------------------------------------------ figures
def forest_means(data, keys, fname, title, ref="native_unmodified"):
    n = len(keys); fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(17, 3.1 * np.ceil(n / 4)), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        nm, better, fmt = METRICS[k]; labs = [l for l in LABELS if l in data]
        for i, l in enumerate(labs):
            m, ci = boot_ci(col(data, l, k))
            ax.errorbar(m, i, xerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color=COL[l], capsize=2.5, ms=4.5)
        ax.axvline(np.nanmean(col(data, ref, k)), color="#6b7280", ls="--", lw=.8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels([NICE[l] if ax in axes[:, 0] else "" for l in labs], fontsize=7); ax.invert_yaxis()
        arrow = {1: " (higher better)", -1: " (lower better)", 0: ""}[better]
        ax.set_title(nm + arrow, fontsize=8); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)


def forest_effects(data, keys, fname, title, subject="q_learning", refs=("native_unmodified",), extra=None):
    """Standardised paired differences (dz-free: raw difference / pooled SD of the metric across matches) so metrics share an axis; sign flipped so + = better for us."""
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(keys) * len(refs) + 1.6)); y = 0; yt, yl = [], []
    for k in keys:
        nm, better, _ = METRICS[k]
        for r in refs:
            if r not in data: continue
            a, b = col(data, subject, k), col(data, r, k); sd = np.nanstd(np.concatenate([a, b]), ddof=1)
            c = compare(a, b); sgn = better if better else 1
            lo, hi = sorted([sgn * c["paired_ci95"][0] / sd, sgn * c["paired_ci95"][1] / sd])
            ax.errorbar(sgn * c["diff"] / sd, y, xerr=[[sgn * c["diff"] / sd - lo], [hi - sgn * c["diff"] / sd]], fmt="o", color=COL[r], capsize=2.5, ms=4.5)
            yt.append(y); yl.append(f"{nm}  vs {NICE[r]}" if len(refs) > 1 else nm); y += 1
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel(f"{NICE[subject]} minus reference (SDs of metric); right = better", fontsize=7)
    ax.set_title(title, fontsize=9); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)


def fig_state_dependence(sd, fname):
    labs = [l for l in ("q_learning", "hand_coach", "q_learning_shuffled_obs", "random_switching") if l in sd]
    fig, axes = plt.subplots(1, len(labs) + 1, figsize=(4.2 * (len(labs) + 1), 3.6))
    zl, pl = ["our third", "middle", "their third"], ["we own", "opp owns", "loose"]
    for ax, l in zip(axes, labs):
        g = np.array(sd[l]["by_zone_poss"])                               # zone x poss x action
        mat = g[:, :, 0] - g[:, :, 2]                                      # P(compact) - P(stretched)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-.6, vmax=.6)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "/".join(f"{int(round(100 * v))}" for v in g[i, j]), ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(pl, fontsize=7); ax.set_yticks(range(3)); ax.set_yticklabels(zl if ax is axes[0] else [""] * 3, fontsize=7)
        ax.set_title(f"{NICE[l]}\nNMI excess {sd[l]['nmi_excess']:.3f}", fontsize=8)
    ax = axes[-1]; xs = np.arange(len(labs)); ax.bar(xs, [sd[l]["nmi_excess"] for l in labs], color=[COL[l] for l in labs])
    ax.set_xticks(xs); ax.set_xticklabels([NICE[l].replace(" ", "\n", 1) for l in labs], fontsize=6.5); ax.set_title("action-state dependence\n(NMI above permutation null)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cells: % compact / default / stretched chosen in each (ball zone x ownership) cell; colour = %compact - %stretched", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)


def fig_recovery(data, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    for l in LABELS:
        if l not in data: continue
        d = [x[1] for ev in data[l]["events"] for x in ev["oop"]]; c = [x[2] for ev in data[l]["events"] for x in ev["oop"]]
        xs, ys = km(d, c); ax.step(xs, ys, where="post", color=COL[l], label=NICE[l], lw=1.4)
    ax.set_xlim(0, 400); ax.set_xlabel("engine steps since we lost the ball"); ax.set_ylabel("P(still without the ball)"); ax.legend(fontsize=7)
    ax.set_title("Time to regain possession (Kaplan-Meier, censored at match end)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / fname, dpi=150); plt.close(fig)


# ------------------------------------------------------------------ final analysis
def final_analysis(data, spec):
    prim = spec["primary"]["metric"]; sec = spec["secondary_metrics"]; labs = [l for l in LABELS if l in data]
    n = len(data["native_unmodified"]["rows"]); seeds = data["native_unmodified"]["seeds"]
    for l in labs: assert data[l]["seeds"] == seeds, "seed sets differ between policies"
    out = {"n_matches_per_policy": n, "seed_range": [seeds[0], seeds[-1]], "spec_sha256": spec["_sha256"], "primary_metric": prim, "means": {}, "comparisons": {}, "pooled_recovery": {}}
    for k in METRICS:
        out["means"][k] = {l: dict(zip(("mean", "ci95"), boot_ci(col(data, l, k)))) | {"sd": float(np.nanstd(col(data, l, k), ddof=1)), "n_nan": int(np.isnan(col(data, l, k)).sum())} for l in labs}
    best_dev = spec["strongest_fixed_baseline_selected_on_dev"]                    # pre-specified
    for k in METRICS:
        nm, better, _ = METRICS[k]; fixed = ["native_unmodified", "fixed_compact", "fixed_stretched"]
        best_test = (max if better >= 0 else min)(fixed, key=lambda l: np.nanmean(col(data, l, k)))      # selected ON TEST: favours the baseline
        refs = {"native": "native_unmodified", "strongest_fixed_dev": best_dev, "strongest_fixed_test_selected": best_test, "hand_coach": "hand_coach",
                "random_switching": "random_switching", "q_shuffled_obs": "q_learning_shuffled_obs"}
        out["comparisons"][k] = {name: compare(col(data, "q_learning", k), col(data, r, k)) | {"ref": r} for name, r in refs.items() if r in data}
    # pooled time-to-regain (cluster bootstrap over matches, paired by seed)
    for key, open_only in (("t_recover_pooled", False), ("t_recover_open_pooled", True)):
        out["pooled_recovery"][key] = {}
        for l in labs:
            out["pooled_recovery"][key][l] = {"mean": pooled_recovery(data, l, range(n), open_only)}
        for r in ("native_unmodified", "fixed_compact", "fixed_stretched", "q_learning_shuffled_obs"):
            if r not in data: continue
            ds = np.array([pooled_recovery(data, "q_learning", ix, open_only) - pooled_recovery(data, r, ix, open_only) for ix in (RNG.integers(0, n, n) for _ in range(1500))])
            out["pooled_recovery"][key][f"q_minus_{r}"] = {"diff": pooled_recovery(data, "q_learning", range(n), open_only) - pooled_recovery(data, r, range(n), open_only), "ci95": [float(np.nanquantile(ds, .025)), float(np.nanquantile(ds, .975))]}
    # Holm over the secondary family, Q vs native
    ps = np.array([out["comparisons"][k]["native"]["p_signflip"] for k in sec]); adj = holm(ps)
    for k, a in zip(sec, adj): out["comparisons"][k]["native"]["p_holm_secondary_family"] = float(a)
    out["state_dependence"] = state_dependence(data); out["q_audit"] = q_audit(data)
    out["manipulation_check_compact_minus_stretched"] = {k: compare(col(data, "fixed_compact", k), col(data, "fixed_stretched", k)) for k in METRICS}
    out["sensitivity"] = sensitivity(data, best_dev)
    return out


SENS = {"min_spell=1": ({"min_spell": 1}, ("recoveries_open", "t_recover_mean", "opp_def_third_entries", "opp_box_entries")),
        "min_spell=3": ({"min_spell": 3}, ("recoveries_open", "t_recover_mean", "opp_def_third_entries", "opp_box_entries")),
        "on-target half-width 0.044 (strict)": ({"post": 0.044}, ("opp_shot_attempts", "opp_shots_on_target")),
        "on-target half-width 0.10 (loose)": ({"post": 0.10}, ("opp_shot_attempts", "opp_shots_on_target")),
        "kick_dv=0.020": ({"kick_dv": 0.02}, ("opp_shot_attempts", "opp_shots_on_target")),
        "kick_dv=0.035": ({"kick_dv": 0.035}, ("opp_shot_attempts", "opp_shots_on_target"))}


def sensitivity(data, ref):
    """Q minus reference under alternative metric-definition parameters (does a conclusion depend on a threshold?)."""
    res = {}
    for name, (over, keys) in SENS.items():
        p = dict(sm.P, **over); vals = {l: [sm.compute_match(tr, p)[0] for tr in data[l]["traces"]] for l in ("q_learning", ref, "native_unmodified")}
        res[name] = {}
        for k in keys:
            q = np.array([m[k] for m in vals["q_learning"]], float)
            res[name][k] = {r: compare(q, np.array([m[k] for m in vals[r]], float)) for r in {"native_unmodified", ref}}
    return res


def md_tables(out, data, spec):
    labs = [l for l in LABELS if l in data]; L = []
    L.append(f"Confirmatory evaluation: {out['n_matches_per_policy']} matches per policy, seeds {out['seed_range'][0]}-{out['seed_range'][1]}, identical seeds for every policy. Spec sha256 `{out['spec_sha256'][:16]}`.")
    L.append(f"\n**Pre-specified primary outcome:** {METRICS[out['primary_metric']][0]} (`{out['primary_metric']}`), Q-learning vs native AI. Everything else is secondary/exploratory.\n")
    groups = [("Scoreboard", ["goals_against", "goals_for", "goal_diff"]), ("Possession & loose ball", ["poss_carry_share", "poss_owned_share", "frac_own", "frac_opp", "frac_loose"]),
              ("Recoveries", ["recoveries_open", "recoveries_all", "recoveries_direct", "losses_all", "t_recover_mean", "t_recover_open_mean"]),
              ("Opponent territorial progression", ["opp_def_third_steps", "opp_def_third_entries", "opp_box_entries", "opp_box_steps", "opp_spell_min_x_mean", "ball_x_mean"]),
              ("Shot-like events (PROXY)", ["opp_shot_attempts", "opp_shots_on_target", "own_shot_attempts", "own_shots_on_target"])]
    for g, keys in groups:
        L.append(f"\n### {g}\n\n| Metric | " + " | ".join(NICE[l] for l in labs) + " |\n|---|" + "---|" * len(labs))
        for k in keys:
            nm, better, fmt = METRICS[k]
            cells = []
            for l in labs:
                m = out["means"][k][l]; cells.append(fmt.format(m["mean"]) + f" [{fmt.format(m['ci95'][0])}, {fmt.format(m['ci95'][1])}]")
            L.append(f"| {nm}{' *(primary)*' if k == out['primary_metric'] else ''} | " + " | ".join(cells) + " |")
    L.append("\n### Trained Q-learning coach minus reference (paired by seed; 95% CI; sign-flip p; **bold** = CI excludes 0)\n")
    L.append("| Metric (better =) | vs Native AI | vs strongest fixed (dev-selected: `" + spec["strongest_fixed_baseline_selected_on_dev"] + "`) | vs strongest fixed (test-selected, favours baseline) | vs Q shuffled obs. (state-dependence) |\n|---|---|---|---|---|")
    for k, (nm, better, fmt) in METRICS.items():
        arrow = {1: "higher", -1: "lower", 0: "descriptive"}[better]; cells = []
        for name in ("native", "strongest_fixed_dev", "strongest_fixed_test_selected", "q_shuffled_obs"):
            c = out["comparisons"][k].get(name)
            if not c: cells.append("-"); continue
            lo, hi = c["paired_ci95"]; sig = (lo > 0 or hi < 0)
            s = f"{c['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={c['p_signflip']:.3f}"
            cells.append(f"**{s}**" if sig else s)
        L.append(f"| {nm} ({arrow}){' *primary*' if k == out['primary_metric'] else ''} | " + " | ".join(cells) + " |")
    L.append("\n### Time to regain possession, pooled over all out-of-possession periods (cluster bootstrap over matches)\n")
    L.append("| Policy | mean steps (all periods) | mean steps (open-play periods) |\n|---|---|---|")
    for l in labs:
        L.append(f"| {NICE[l]} | {out['pooled_recovery']['t_recover_pooled'][l]['mean']:.1f} | {out['pooled_recovery']['t_recover_open_pooled'][l]['mean']:.1f} |")
    for r in ("native_unmodified", "fixed_compact", "fixed_stretched", "q_learning_shuffled_obs"):
        a, b = out["pooled_recovery"]["t_recover_pooled"].get(f"q_minus_{r}"), out["pooled_recovery"]["t_recover_open_pooled"].get(f"q_minus_{r}")
        if a: L.append(f"| Q minus {NICE[r]} | {a['diff']:+.1f} [{a['ci95'][0]:+.1f}, {a['ci95'][1]:+.1f}] | {b['diff']:+.1f} [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}] |")
    sd = out["state_dependence"]
    L.append("\n### Is the Q policy state-dependent? (actions vs (ball zone, ownership, score, phase), 81 states)\n\n| Policy | decisions | % compact / default / stretched | NMI | permutation-null NMI | excess | p |\n|---|---|---|---|---|---|---|")
    for l in labs:
        s = sd[l]; L.append(f"| {NICE[l]} | {s['n_decisions']} | {' / '.join(f'{100 * v:.0f}' for v in s['action_frac_compact_default_stretched'])} | {s['nmi']:.3f} | {s['nmi_null']:.3f} | {s['nmi_excess']:+.3f} | {s['p_perm']:.3f} |")
    L.append("\nQ-table audit: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in out["q_audit"].items()))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=("dev", "final"), required=True); args = ap.parse_args()
    FIGS.mkdir(exist_ok=True)
    data = load(args.set)
    if args.set == "dev":
        out = dev_diagnostics(data)
        json.dump(out, open(R / "stats_dev_diagnostics.json", "w"), indent=1)
        keys = [k for k in METRICS if not k.startswith("own_") and k not in ("frac_own", "frac_opp", "recoveries_direct", "goals_for")]
        forest_means(data, keys, "stats_dev_means.png", f"DEVELOPMENT diagnostics (n={out['n_matches_per_policy']} matches/policy; not confirmatory)")
        fig_state_dependence(out["state_dependence"], "stats_dev_state_dependence.png"); fig_recovery(data, "stats_dev_recovery_km.png")
        print(json.dumps({k: out[k] for k in ("shot_proxy", "owned_run_length", "recoveries_open_native_by_min_spell", "q_audit")}, indent=1))
        return
    sp = Path("eval_spec.json"); raw = sp.read_bytes(); spec = json.loads(raw); spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    meta = json.load(open(R / "stats_final" / "meta.json"))
    assert meta.get("spec_sha256") == spec["_sha256"], "eval_spec.json changed after the final matches were collected: refusing to analyse"
    out = final_analysis(data, spec)
    json.dump(out, open(R / "stats_final_analysis.json", "w"), indent=1)
    (R / "stats_final_tables.md").write_text(md_tables(out, data, spec))
    keys = [k for k in METRICS if k not in ("frac_own", "frac_opp", "frac_loose")]
    forest_means(data, [k for k in keys if not k.startswith("own_") and k != "goals_for"], "stats_final_means.png", f"Confirmatory evaluation, n={out['n_matches_per_policy']} matches/policy (mean, 95% bootstrap CI)")
    forest_effects(data, [k for k in keys if k != "goals_for" and not k.startswith("own_")], "stats_final_effects_vs_native_fixed.png",
                   "Q-learning minus reference, standardised; right = better", refs=("native_unmodified", spec["strongest_fixed_baseline_selected_on_dev"]))
    forest_effects(data, [k for k in keys if k != "goals_for" and not k.startswith("own_")], "stats_final_effects_vs_shuffled.png", "Q-learning minus Q-learning with shuffled observations (state-dependence test)", refs=("q_learning_shuffled_obs",))
    fig_state_dependence(out["state_dependence"], "stats_final_state_dependence.png"); fig_recovery(data, "stats_final_recovery_km.png")
    print((R / "stats_final_tables.md").read_text())


if __name__ == "__main__":
    main()

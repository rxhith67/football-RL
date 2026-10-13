"""Analyse paired native-depth rollouts. Runs on Windows or WSL (numpy only).
usage: python analyze_stage2.py dev|held [dev_best_json]
Pre-declared primary metric: opponent progress toward our goal before first recovery (lower = better for us).
Secondary: recovery time (censored at horizon+1), dangerous-ball entries, goals conceded."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent / "results"
tag = sys.argv[1]
rng = np.random.default_rng(7)
rows = []
for f in sorted(glob.glob(str(here / f"{tag}_rep*.json"))):
    rows += json.load(open(f))["rows"]
SET = sorted({r["depth"] for r in rows})
METRICS = {"opp_progress": 1, "recovery_censored": 1, "dangerous_entries": 1, "goals_conceded": 1,
           "back_line_x": 1, "team_length": 1}

def val(r, m):
    if m == "back_line_x": return r["shape_opp_possession"]["back_line_x"]
    if m == "team_length": return r["shape_opp_possession"]["team_length"]
    return r[m]

# per (set,state,setting): mean across replicates
cell = defaultdict(lambda: defaultdict(list))
cat = {}
for r in rows:
    key = (r["set"], r["state_id"])
    cat[key] = r["category"]
    for m in METRICS:
        v = val(r, m)
        if v is not None: cell[(key, r["depth"])][m].append(v)
keys = sorted(cat)
def mean_cell(key, dep, m):
    v = cell[(key, dep)][m]; return float(np.mean(v)) if v else np.nan
def boot(x, B=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return float(x.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], len(x)

out = {"n_states": len(keys), "replicates": sorted({r["replicate"] for r in rows}), "settings": SET}
# noise: SD across replicates for identical state+setting (engine has process-level modes)
noise = {m: float(np.nanmean([np.std(cell[(k, d)][m]) for k in keys for d in SET if len(cell[(k, d)][m]) > 1])) for m in METRICS}
out["mean_between_replicate_sd"] = noise
# 1. effect of each setting vs default (-1), paired by state
eff = {}
for d in SET:
    if d == -1.0: continue
    eff[str(d)] = {}
    for m in METRICS:
        diffs = [mean_cell(k, d, m) - mean_cell(k, -1.0, m) for k in keys]
        mean, ci, n = boot(diffs)
        eff[str(d)][m] = {"mean_diff_vs_default": mean, "ci95_over_states": ci}
out["effect_vs_default_all_states"] = eff
# 2. by category
bycat = {}
for c in sorted(set(cat.values())):
    ks = [k for k in keys if cat[k] == c]
    bycat[c] = {"n": len(ks)}
    for d in SET:
        bycat[c][str(d)] = {m: float(np.nanmean([mean_cell(k, d, m) for k in ks])) for m in ("opp_progress", "recovery_censored", "dangerous_entries", "goals_conceded")}
    prog = {d: bycat[c][str(d)]["opp_progress"] for d in SET}
    bycat[c]["preferred_by_primary_metric"] = min(prog, key=prog.get)
    rec = {d: bycat[c][str(d)]["recovery_censored"] for d in SET}
    bycat[c]["preferred_by_recovery_time"] = min(rec, key=rec.get)
out["by_category"] = bycat
out["pooled_means"] = {str(d): {m: float(np.nanmean([mean_cell(k, d, m) for k in keys])) for m in METRICS} for d in SET}
json.dump(out, open(here / f"analysis_{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# 3. state-dependent policy test: needs the dev-best-by-category choice (from dev) applied on this set
if len(sys.argv) > 2:
    dev = json.load(open(sys.argv[2]))
    pref = {c: float(dev["by_category"][c]["preferred_by_primary_metric"]) for c in dev["by_category"]}
    best_fixed = min(SET, key=lambda d: dev["pooled_means"][str(d)]["opp_progress"])
    pol, fix, dflt = [], [], []
    for k in keys:
        pol.append(mean_cell(k, pref[cat[k]], "opp_progress")); fix.append(mean_cell(k, best_fixed, "opp_progress")); dflt.append(mean_cell(k, -1.0, "opp_progress"))
    pol, fix, dflt = map(np.array, (pol, fix, dflt))
    res = {"dev_preference_by_category": pref, "dev_best_fixed": best_fixed,
           "category_policy_minus_default": boot(pol - dflt), "best_fixed_minus_default": boot(fix - dflt),
           "category_policy_minus_best_fixed": boot(pol - fix)}
    out["held_out_state_dependent_test"] = res
    json.dump(out, open(here / f"analysis_{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

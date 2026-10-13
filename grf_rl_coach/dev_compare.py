"""Preliminary DEVELOPMENT comparison (diagnostic, not the final evaluation). Seeds 700000+ are independent of
training (100000+), curve evaluation (500000+) and the final test (900000+). Results are cached per label so that
later checkpoints can be added without rerunning baselines.
usage: bash run.sh dev_compare.py --episodes 30 --workers 6 --q q41=checkpoints/q_it41_frozen.npz [--q final=...]"""
import argparse
import json
from pathlib import Path

import numpy as np

from rollout import Runner

DEV_SEED0 = 700000
OUT = Path("results/dev_compare.json")


def boot(d, rng, B=10000):
    d = np.asarray(d, float)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--q", action="append", default=[], help="label=path to a Q checkpoint (.npz)")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    seeds = [DEV_SEED0 + i for i in range(args.episodes)]
    store = json.load(open(OUT)) if OUT.exists() else {"seeds": seeds, "rows": {}}
    assert store["seeds"] == seeds, "cached results use a different seed set"
    rows = store["rows"]
    runner = Runner(args.workers)
    pools = {}

    def run(label, **kw):
        if label in rows:
            print("cached", label); return
        res = runner.run([dict(seed=s, label=label, **kw) for s in seeds])
        rows[label] = [e for e, _ in res]
        pools[label] = [t[1] for _, tr in res for t in tr]
        gd = np.array([e["goal_diff"] for e in rows[label]])
        dec = sorted({e["decisions"] for e in rows[label]})
        print(f"{label:28s} goal_diff {gd.mean():+.2f} (SD {gd.std(ddof=1):.2f})  decisions/match {dec}", flush=True)
        json.dump(store, open(OUT, "w"))

    if not args.no_baselines:
        run("native_unmodified", policy="native")
        run("fixed_compact", policy="fixed", action=0)
        run("fixed_stretched", policy="fixed", action=2)
        run("hand_coach", policy="hand")
        run("random_switching", policy="random")
    for spec in args.q:
        label, path = spec.split("=", 1)
        q = np.load(path)["q"]
        run(label, policy="q", q=q, epsilon=0.0, collect=True)
        if label in pools:                                            # pool exists only when freshly run
            run(label + "_shuffled_obs", policy="q", q=q, epsilon=0.0, shuffle_pool=pools[label])
        elif label + "_shuffled_obs" not in rows:
            raise SystemExit("run the Q label fresh to build its shuffle pool")
    runner.close()

    rng = np.random.default_rng(1)
    print("\nlabel                         GD mean ± SE   W/D/L   return   actions C/D/S    paired GD vs native [95% CI]")
    base = np.array([e["goal_diff"] for e in rows["native_unmodified"]], float)
    summary = {}
    for l, eps in rows.items():
        gd = np.array([e["goal_diff"] for e in eps], float)
        ac = np.sum([e["action_counts"] for e in eps], 0); fr = ac / ac.sum()
        w, d_ = int((gd > 0).sum()), int((gd == 0).sum())
        m, ci = boot(gd - base, rng)
        summary[l] = {"goal_diff_mean": float(gd.mean()), "se": float(gd.std(ddof=1) / np.sqrt(len(gd))), "wdl": [w, d_, len(gd) - w - d_],
                      "return_mean": float(np.mean([e["return"] for e in eps])), "action_frac": fr.round(3).tolist(),
                      "paired_vs_native": {"mean": m, "ci95": ci}, "decisions_per_match": sorted({e["decisions"] for e in eps})}
        print(f"{l:28s} {gd.mean():+.2f} ± {gd.std(ddof=1) / np.sqrt(len(gd)):.2f}   {w}-{d_}-{len(gd) - w - d_}   {summary[l]['return_mean']:+.2f}   "
              f"{'/'.join(f'{x:.2f}' for x in fr)}   {m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    store["summary"] = summary
    json.dump(store, open(OUT, "w"))


if __name__ == "__main__":
    main()

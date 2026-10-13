"""Sanity-check metrics.py on the calibration matches: sample values, flicker removal, shot-detector recall on goals."""
import pickle
import numpy as np
import metrics as M

res = pickle.load(open("results/calib/native_unmodified.pkl", "rb"))
allm = [M.match_metrics(st) for ep, st, dec in res]
for k in allm[0]:
    v = np.array([m[k] for m in allm], float)
    print(f"{k:34s} mean {np.nanmean(v):8.3f}  min {np.nanmin(v):8.3f}  max {np.nanmax(v):8.3f}")
# flicker: raw owner flips vs debounced switches
raw_flips, deb = 0, 0
for ep, st, dec in res:
    o = st[:, 0]; raw_flips += int(((o[1:] != o[:-1]) & (o[1:] >= 0) & (o[:-1] >= 0)).sum()); deb += len(M.possessor_events(st[:, 0], st[:, 5], st[:, 6], st[:, 7]))
print("direct owner flips between teams (raw):", raw_flips, "| debounced possession switches:", deb)
# shot detector recall on goals
hit = tot = 0; hit_ot = 0
for ep, st, dec in res:
    for t in range(1, len(st)):
        for team, col in ((0, 6), (1, 7)):
            if st[t, col] > st[t - 1, col]:
                tot += 1
                lo = max(0, t - 60); sub = st[lo:t + 1]
                n, ot = M.shots(sub, team)
                hit += n > 0; hit_ot += ot > 0
print(f"goals {tot}: shot-like event detected before {hit} ({hit/tot:.0%}); flagged on target {hit_ot} ({hit_ot/tot:.0%})")

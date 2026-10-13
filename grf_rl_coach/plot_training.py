"""Learning curves from logs/*.jsonl.  usage: python plot_training.py q_main [dqn_main]"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tags = sys.argv[1:]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for tag, col in zip(tags, ("#1b6b45", "#e0a526")):
    L = [json.loads(l) for l in Path(f"logs/{tag}.jsonl").read_text().splitlines()]
    ep = np.array([r["episodes"] for r in L]); ret = np.array([r["train_return_mean"] for r in L])
    k = 5
    sm = np.convolve(ret, np.ones(k) / k, mode="valid")
    ax[0].plot(ep, ret, color=col, alpha=.25); ax[0].plot(ep[k - 1:], sm, color=col, label=tag)
    dv = [(r["episodes"], r["dev_goal_diff_mean"]) for r in L if "dev_goal_diff_mean" in r]
    ax[1].plot(*zip(*dv), "o-", color=col, label=tag)
    fr = np.array([r["train_action_frac"] for r in L])
    for j, st in enumerate(("-", "--", ":")):
        ax[2].plot(ep, fr[:, j], st, color=col, label=f"{tag} {('compact','default','stretched')[j]}")
ax[0].set_title("Training return per match (incl. exploration)"); ax[0].set_xlabel("training matches"); ax[0].legend()
ax[1].set_title("Greedy policy: goal diff on 6 dev seeds"); ax[1].set_xlabel("training matches"); ax[1].legend()
ax[2].set_title("Training action fractions"); ax[2].set_xlabel("training matches"); ax[2].legend(fontsize=6)
plt.tight_layout(); Path("figs").mkdir(exist_ok=True); plt.savefig(f"figs/training_{'_'.join(tags)}.png", dpi=160)
print("saved")

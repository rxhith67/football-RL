"""Plot state-level GRF recovery outcomes; each marker is one rollout."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    policies = ("normal", "press", "cover", "adaptive")
    colors = {"normal": "#222222", "press": "#e07a26",
              "cover": "#2567a5", "adaptive": "#7b4ba8"}
    offsets = {"normal": -0.24, "press": -0.08,
               "cover": 0.08, "adaptive": 0.24}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, category in zip(axes, ("midfield", "penalty", "wide")):
        rows = [r for r in data["results"] if r["category"] == category]
        states = sorted({r["state_id"] for r in rows})
        for index, state in enumerate(states):
            for policy in policies:
                row = next(r for r in rows if r["state_id"] == state
                           and r["policy"] == policy)
                recovered = row["time_to_recovery"] is not None
                y = row["time_to_recovery"] if recovered else 205
                marker = "o" if recovered else "^"
                x = index + offsets[policy]
                ax.scatter(x, y, s=60, marker=marker,
                           color=colors[policy] if recovered else "white",
                           edgecolor=colors[policy], linewidth=1.7,
                           zorder=3)
                if row["goals_conceded"]:
                    ax.text(x, y + 8, "*", color="#b21925", ha="center",
                            va="center", fontsize=14, fontweight="bold")
        ax.axhline(200, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(states)),
                      [state.rsplit("_", 1)[-1] for state in states])
        ax.set_xlim(-0.55, len(states) - 0.45)
        ax.set_ylim(0, 225)
        ax.set_title(category.title())
        ax.set_xlabel("Match-state index")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Steps until blue recovers possession")
    fig.suptitle("Paired GRF defensive states · 200-step horizon", y=1.02)
    handles = [plt.Line2D([], [], marker="o", linestyle="None",
                          markerfacecolor=colors[p], markeredgecolor=colors[p],
                          label=p.title()) for p in policies]
    handles += [plt.Line2D([], [], marker="^", linestyle="None",
                           markerfacecolor="white", markeredgecolor="#555555",
                           label="No recovery by 200")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.text(0.99, 0.01, "* goal conceded within 200 steps",
             ha="right", fontsize=9, color="#b21925")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()

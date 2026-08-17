"""Render a match to an animation — the evidence for Gate 1's human judgement.

The first Gate 1 condition is not a number: *a football-literate person watching
this says "yes, that looks like football", unprompted*. That judgement needs
something to watch, and it is the one condition that can send the project to
Option B, so it deserves a proper render rather than a scatter of dots.

Usage::

    python scripts/animate_match.py --seed 3 --out figures/match.gif
    python scripts/animate_match.py --seed 3 --minutes 10 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pitchsim import Tactic, load_replay  # noqa: E402
from pitchsim.engine import PITCH_L, PITCH_W, PitchSimEngine  # noqa: E402

PITCH_GREEN = "#2f7a3f"
LINE = "#e8f0e8"


def draw_pitch(ax) -> None:
    """Markings, so the eye can judge shape and distance rather than dot soup."""
    ax.set_facecolor(PITCH_GREEN)
    ax.set_xlim(-3, PITCH_L + 3)
    ax.set_ylim(-3, PITCH_W + 3)
    ax.set_aspect("equal")
    ax.axis("off")

    kw = dict(color=LINE, linewidth=1.4, zorder=1)
    ax.plot([0, PITCH_L, PITCH_L, 0, 0], [0, 0, PITCH_W, PITCH_W, 0], **kw)
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], **kw)

    circle = plt.Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fill=False, **kw)
    ax.add_patch(circle)

    for x0, sign in ((0.0, 1.0), (PITCH_L, -1.0)):
        # 18-yard box
        ax.plot([x0, x0 + sign * 16.5, x0 + sign * 16.5, x0],
                [13.85, 13.85, PITCH_W - 13.85, PITCH_W - 13.85], **kw)
        # 6-yard box
        ax.plot([x0, x0 + sign * 5.5, x0 + sign * 5.5, x0],
                [24.85, 24.85, PITCH_W - 24.85, PITCH_W - 24.85], **kw)
        # goal
        ax.plot([x0, x0 - sign * 1.5, x0 - sign * 1.5, x0],
                [30.34, 30.34, 37.66, 37.66], color=LINE, linewidth=2.2, zorder=1)


def build_replay(seed: int, minutes: float, home: Tactic, away: Tactic) -> dict:
    sim = PitchSimEngine()
    sim.reset(seed=seed, home_tactic=home, away_tactic=away, record=True)
    while not sim.done and sim.state.minute < minutes:
        sim.advance(5.0)
    replay = sim.replay()
    result = sim.result()
    print(f"score {result.score[0]}-{result.score[1]}  shots {result.shots}  "
          f"xG {result.xg[0]:.2f}-{result.xg[1]:.2f}  "
          f"possession {result.possession_home:.0%}  ticks {len(replay['ticks'])}")
    return replay


def animate(replay: dict, out: Path | None, show: bool, fps: int = 20) -> None:
    import matplotlib.animation as animation

    ticks = replay["ticks"]
    if not ticks:
        raise SystemExit("replay has no ticks — was record=True set?")

    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.patch.set_facecolor("#12261a")
    draw_pitch(ax)

    home_dots = ax.scatter([], [], s=115, c="#e63946", edgecolors="white",
                           linewidths=1.1, zorder=3, label="Home")
    away_dots = ax.scatter([], [], s=115, c="#1d6fd0", edgecolors="white",
                           linewidths=1.1, zorder=3, label="Away")
    ball_dot = ax.scatter([], [], s=52, c="#ffffff", edgecolors="black",
                          linewidths=1.0, zorder=4)
    clock = ax.text(PITCH_L / 2, PITCH_W + 1.2, "", ha="center", va="bottom",
                    color="white", fontsize=13, family="monospace", zorder=5)

    goals = sorted(replay.get("events", []), key=lambda e: e["minute"])

    def update(frame):
        tick = ticks[frame]
        home_dots.set_offsets(np.array(tick["home"]))
        away_dots.set_offsets(np.array(tick["away"]))
        ball_dot.set_offsets(np.array([tick["ball"]]))

        minute = tick["t"] / 60.0
        scored = [g for g in goals if g["minute"] <= minute and g["type"] == "goal"]
        home_goals = sum(1 for g in scored if g["side"] == 0)
        away_goals = sum(1 for g in scored if g["side"] == 1)
        clock.set_text(f"{int(minute):02d}:{int(tick['t'] % 60):02d}    "
                       f"{home_goals} - {away_goals}")
        return home_dots, away_dots, ball_dot, clock

    anim = animation.FuncAnimation(fig, update, frames=len(ticks),
                                   interval=1000 / fps, blit=True)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = "pillow" if out.suffix == ".gif" else "ffmpeg"
        print(f"writing {len(ticks)} frames -> {out} ...")
        anim.save(out, writer=writer, fps=fps, dpi=80)
        print("done")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a PITCHSIM match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "match.gif")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--replay", type=Path, help="animate a saved replay JSON instead")
    parser.add_argument("--home", default="4-4-2")
    parser.add_argument("--away", default="4-4-2")
    args = parser.parse_args()

    replay = (load_replay(args.replay) if args.replay
              else build_replay(args.seed, args.minutes,
                                Tactic.from_names(formation=args.home),
                                Tactic.from_names(formation=args.away)))
    animate(replay, None if args.show else args.out, args.show)


if __name__ == "__main__":
    import matplotlib

    if "--show" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    main()
else:
    import matplotlib.pyplot as plt  # noqa: E402

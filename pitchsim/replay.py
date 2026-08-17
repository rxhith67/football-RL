"""Versioned replay JSON — the contract between the sim and the console.

Frozen in Phase 0 so owner C can build the broadcast pitch, belief bars and xG
race against the stub while owner A is still writing physics. The console reads
only this format; it never imports the engine.

Layout (schema version 1)::

    {
      "schema_version": 1,
      "meta":    {"seed": int, "home": str, "away": str, "dt": float, ...},
      "ticks":   [{"t": float, "home": [[x, y], ...], "away": [...], "ball": [x, y]}],
      "epochs":  [{"minute": float, "side": 0|1, "tactic": {...}, "belief": [...],
                   "trigger": str, "value": float|null}],
      "events":  [{"minute": float, "type": str, "side": 0|1, ...}],
      "traces":  {"xg_home": [[minute, cumulative], ...], "xg_away": [...],
                  "win_prob_home": [[minute, p], ...]},
      "result":  {...MatchResult...}
    }

Positions are in metres on a 105 x 68 pitch, home attacking +x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPLAY_SCHEMA_VERSION = 1

#: Tick sampling for the console. The engine runs at dt = 0.2 s, but the browser
#: only needs ~5 fps of animation, so record every Nth tick to keep replay JSON
#: to a few MB per match rather than tens.
DEFAULT_TICK_STRIDE = 5


def _round(a, nd: int = 2) -> list:
    return np.round(np.asarray(a, dtype=np.float32), nd).tolist()


@dataclass
class ReplayBuilder:
    """Accumulates a replay and serialises it to the frozen schema."""

    seed: int | None = None
    home_name: str = "GAFFER"
    away_name: str = "Opponent"
    dt: float = 0.2
    tick_stride: int = DEFAULT_TICK_STRIDE

    ticks: list[dict[str, Any]] = field(default_factory=list)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    xg_home: list[list[float]] = field(default_factory=list)
    xg_away: list[list[float]] = field(default_factory=list)
    win_prob_home: list[list[float]] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def add_tick(self, t: float, home_xy, away_xy, ball_xy) -> None:
        self.ticks.append(
            {"t": round(float(t), 2), "home": _round(home_xy),
             "away": _round(away_xy), "ball": _round(ball_xy)}
        )

    def add_epoch(
        self,
        minute: float,
        side: int,
        tactic: dict[str, str],
        belief: list[float] | None = None,
        trigger: str = "scheduled",
        value: float | None = None,
        top_features: list[tuple[str, float]] | None = None,
    ) -> None:
        """One coaching decision.

        `belief`, `value` and `top_features` drive the "Manager's bench" and
        "Opponent read" panels; they are optional so the stub can omit them.
        """
        self.epochs.append(
            {"minute": round(float(minute), 2), "side": int(side), "tactic": tactic,
             "belief": None if belief is None else _round(belief, 4),
             "trigger": trigger,
             "value": None if value is None else round(float(value), 4),
             "top_features": top_features}
        )

    def add_event(self, minute: float, type_: str, side: int, **extra: Any) -> None:
        self.events.append(
            {"minute": round(float(minute), 2), "type": type_, "side": int(side), **extra}
        )

    def add_xg(self, minute: float, home_cum: float, away_cum: float) -> None:
        self.xg_home.append([round(float(minute), 2), round(float(home_cum), 4)])
        self.xg_away.append([round(float(minute), 2), round(float(away_cum), 4)])

    def add_win_prob(self, minute: float, p_home: float) -> None:
        self.win_prob_home.append([round(float(minute), 2), round(float(p_home), 4)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "meta": {"seed": self.seed, "home": self.home_name, "away": self.away_name,
                     "dt": self.dt, "tick_stride": self.tick_stride,
                     "pitch": [105.0, 68.0]},
            "ticks": self.ticks,
            "epochs": self.epochs,
            "events": self.events,
            "traces": {"xg_home": self.xg_home, "xg_away": self.xg_away,
                       "win_prob_home": self.win_prob_home},
            "result": self.result,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict()), encoding="utf-8")
        return path


def validate_replay(replay: dict[str, Any]) -> list[str]:
    """Structural check. Returns a list of problems; empty means valid.

    Used by the console's test suite so a schema drift fails loudly on the
    producing side rather than silently rendering an empty pitch.
    """
    problems: list[str] = []
    if replay.get("schema_version") != REPLAY_SCHEMA_VERSION:
        problems.append(
            f"schema_version {replay.get('schema_version')} != {REPLAY_SCHEMA_VERSION}"
        )
    for key in ("meta", "ticks", "epochs", "events", "traces"):
        if key not in replay:
            problems.append(f"missing top-level key: {key}")

    for i, tick in enumerate(replay.get("ticks", [])[:50]):
        for key in ("t", "home", "away", "ball"):
            if key not in tick:
                problems.append(f"tick {i}: missing {key}")
                break
        else:
            if len(tick["home"]) != 11 or len(tick["away"]) != 11:
                problems.append(f"tick {i}: expected 11 players per side")
            if len(tick["ball"]) != 2:
                problems.append(f"tick {i}: ball must be [x, y]")

    for i, epoch in enumerate(replay.get("epochs", [])):
        if "minute" not in epoch or "tactic" not in epoch:
            problems.append(f"epoch {i}: missing minute or tactic")
    return problems


def load_replay(path: str | Path) -> dict[str, Any]:
    replay = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_replay(replay)
    if problems:
        raise ValueError("invalid replay: " + "; ".join(problems))
    return replay

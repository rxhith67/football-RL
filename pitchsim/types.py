"""Frozen data contract between the environment, the learner and the console.

This module is the single most load-bearing file in the repository. The engine implements PITCHSIM behind these types; the Gymnasium wrapper, the
console and the evaluation harness all build against them.
Because a stub implementation ships in Phase 0 (see `pitchsim.stub`), B and C are
never blocked on the real engine.

Changing any field here breaks all three workstreams at once, so treat additions
as cheap and renames/removals as expensive. `SCHEMA_VERSION` must be bumped on
any change that is not purely additive.

Plan reference: sections 3.2, 3.3 and 6.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any

import numpy as np

#: 3 — replaced redundant shots-against with measured defensive-line height;
#: the opposite side's shots-for already carries the former signal.
SCHEMA_VERSION = 3

# --------------------------------------------------------------------------
# Tactic space — mirrors config/tactics.yaml. MultiDiscrete([5, 3, 3, 3, 6]).
# --------------------------------------------------------------------------

FORMATIONS = ("4-3-3", "4-2-3-1", "3-5-2", "5-4-1", "4-4-2")
BLOCK_HEIGHTS = ("low", "mid", "high")
PRESS_LEVELS = ("contain", "balanced", "gegenpress")
TEMPOS = ("patient", "balanced", "direct")
SUBSTITUTIONS = ("none", "bench_1", "bench_2", "bench_3", "bench_4", "bench_5")

HEAD_SIZES = (
    len(FORMATIONS),
    len(BLOCK_HEIGHTS),
    len(PRESS_LEVELS),
    len(TEMPOS),
    len(SUBSTITUTIONS),
)
MASK_LENGTH = sum(HEAD_SIZES)  # 20 — a FLAT boolean array, per design note 3.3
N_ARCHETYPES = 6


class Side(IntEnum):
    """Which team. The learner is HOME by convention; the sim is symmetric."""

    HOME = 0
    AWAY = 1

    @property
    def other(self) -> "Side":
        return Side.AWAY if self is Side.HOME else Side.HOME


@dataclass(frozen=True, slots=True)
class Tactic:
    """One coaching decision: the five action heads.

    Indices, not strings, so this maps directly onto the MultiDiscrete action.
    """

    formation: int = 0
    block_height: int = 1
    press: int = 1
    tempo: int = 1
    substitution: int = 0

    def __post_init__(self) -> None:
        for value, size, name in zip(self.as_array(), HEAD_SIZES, self._names()):
            if not 0 <= int(value) < size:
                raise ValueError(f"{name} index {value} out of range [0, {size})")

    @staticmethod
    def _names() -> tuple[str, ...]:
        return ("formation", "block_height", "press", "tempo", "substitution")

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.formation, self.block_height, self.press, self.tempo, self.substitution],
            dtype=np.int64,
        )

    @classmethod
    def from_array(cls, arr) -> "Tactic":
        a = np.asarray(arr, dtype=np.int64).ravel()
        if a.size != len(HEAD_SIZES):
            raise ValueError(f"expected {len(HEAD_SIZES)} heads, got {a.size}")
        return cls(*(int(v) for v in a))

    @classmethod
    def from_names(
        cls,
        formation: str = "4-4-2",
        block_height: str = "mid",
        press: str = "balanced",
        tempo: str = "balanced",
        substitution: str = "none",
    ) -> "Tactic":
        return cls(
            FORMATIONS.index(formation),
            BLOCK_HEIGHTS.index(block_height),
            PRESS_LEVELS.index(press),
            TEMPOS.index(tempo),
            SUBSTITUTIONS.index(substitution),
        )

    def as_names(self) -> dict[str, str]:
        return {
            "formation": FORMATIONS[self.formation],
            "block_height": BLOCK_HEIGHTS[self.block_height],
            "press": PRESS_LEVELS[self.press],
            "tempo": TEMPOS[self.tempo],
            "substitution": SUBSTITUTIONS[self.substitution],
        }

    def one_hot(self) -> np.ndarray:
        """14-dim one-hot over the four *style* heads (substitution excluded).

        Substitution is a transient action rather than a persistent style, so it
        is not part of the observed tactic state (design note 3.3).
        """
        out = np.zeros(sum(HEAD_SIZES[:4]), dtype=np.float32)
        offset = 0
        for value, size in zip(self.as_array()[:4], HEAD_SIZES[:4]):
            out[offset + int(value)] = 1.0
            offset += size
        return out

    def without_substitution(self) -> "Tactic":
        return Tactic(self.formation, self.block_height, self.press, self.tempo, 0)

    def style_differs_from(self, other: "Tactic") -> bool:
        """Churn indicator c_k for the reward (design note 3.3)."""
        return self.without_substitution() != other.without_substitution()


# --------------------------------------------------------------------------
# Observable match statistics — the ONLY window onto the opponent's plan.
# --------------------------------------------------------------------------


@dataclass(slots=True)
class WindowStats:
    """Rolling observable statistics for one side over one window.

    These nine fields are the belief filter's entire input (design note 3.4)
    and half of the observation vector. The opponent's *tactic* is never exposed;
    only this statistical footprint is.
    """

    possession: float = 0.5        # share in [0, 1]
    shots_for: float = 0.0         # per window
    defensive_line_height: float = 0.4  # progress from own goal, in [0, 1]
    xg_for: float = 0.0
    xg_against: float = 0.0
    field_tilt: float = 0.5        # share of final-third touches
    ppda: float = 12.0             # opponent passes allowed per defensive action
    pass_completion: float = 0.80
    mean_pass_length: float = 18.0  # metres — the "pass-length profile" signal

    N_FEATURES = 9

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.possession, self.shots_for, self.defensive_line_height, self.xg_for,
             self.xg_against, self.field_tilt, self.ppda, self.pass_completion,
             self.mean_pass_length],
            dtype=np.float32,
        )

    def normalised(self) -> np.ndarray:
        """Scaled to roughly [0, 1] for the policy network."""
        a = self.as_array()
        scale = np.array([1.0, 8.0, 1.0, 1.5, 1.5, 1.0, 30.0, 1.0, 40.0],
                         dtype=np.float32)
        return np.clip(a / scale, 0.0, 1.5)

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        return ("possession", "shots_for", "defensive_line_height", "xg_for",
                "xg_against", "field_tilt", "ppda", "pass_completion",
                "mean_pass_length")


@dataclass(slots=True)
class SquadState:
    """Per-side squad condition. Stamina is 1 = FRESH, 0 = EXHAUSTED.

    Note the convention deliberately inverts GRF's `tired_factor`, whose sign the
    original draft got backwards (design note 1, row 6).
    """

    stamina: np.ndarray = field(default_factory=lambda: np.ones(11, dtype=np.float32))
    yellow_cards: int = 0
    red_cards: int = 0
    subs_used: int = 0
    windows_used: int = 0

    def fatigue_penalty(self, threshold: float = 0.35) -> float:
        """Mean squared stamina deficit below `threshold` — correctly signed."""
        deficit = np.clip(threshold - self.stamina, 0.0, None)
        return float(np.mean(deficit**2))

    def n_below(self, threshold: float = 0.35) -> int:
        return int(np.sum(self.stamina < threshold))


@dataclass(slots=True)
class MatchState:
    """Full match state at a decision epoch."""

    minute: float = 0.0
    score: tuple[int, int] = (0, 0)
    squads: tuple[SquadState, SquadState] = field(
        default_factory=lambda: (SquadState(), SquadState())
    )
    tactics: tuple[Tactic, Tactic] = field(
        default_factory=lambda: (Tactic(), Tactic())
    )
    stats: tuple[WindowStats, WindowStats] = field(
        default_factory=lambda: (WindowStats(), WindowStats())
    )
    cumulative_xg: tuple[float, float] = (0.0, 0.0)
    done: bool = False

    def score_diff(self, side: Side = Side.HOME) -> int:
        return self.score[side] - self.score[side.other]

    def xg_diff(self, side: Side = Side.HOME) -> float:
        return self.cumulative_xg[side] - self.cumulative_xg[side.other]

    def result(self, side: Side = Side.HOME) -> float:
        """Terminal reward: +1 win, 0 draw, -1 loss."""
        d = self.score_diff(side)
        return float(np.sign(d))


@dataclass(slots=True)
class SegmentResult:
    """What the sim returns when advanced between two decision epochs."""

    state: MatchState
    duration_minutes: float          # tau_k — variable, hence a genuine SMDP
    events: list[dict[str, Any]] = field(default_factory=list)
    trigger: str = "scheduled"       # or goal_for / goal_against / red_card / half_time

    def as_dict(self) -> dict[str, Any]:
        return {
            "minute": self.state.minute,
            "score": list(self.state.score),
            "duration_minutes": self.duration_minutes,
            "trigger": self.trigger,
            "events": self.events,
        }


@dataclass(slots=True)
class MatchResult:
    """Terminal summary — the unit of the evaluation harness."""

    score: tuple[int, int]
    xg: tuple[float, float]
    shots: tuple[int, int]
    possession_home: float
    n_decisions: int
    seed: int | None = None
    home_trailing_at_half: bool = False   # feeds the comeback-rate metric

    def points(self, side: Side = Side.HOME) -> int:
        d = self.score[side] - self.score[side.other]
        return 3 if d > 0 else (1 if d == 0 else 0)

    def as_dict(self) -> dict[str, Any]:
        """Plain Python types only.

        NumPy scalars survive `asdict` and then blow up any JSON encoder that is
        stricter than `json` itself — FastAPI rejects `numpy.bool_` outright. The
        contract promises JSON-serialisable output, so coerce here rather than
        making every consumer defend itself.
        """
        def plain(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        return {key: plain(value) for key, value in asdict(self).items()}

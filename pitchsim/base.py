"""The PITCHSIM interface. Implemented twice: by the stub (Phase 0) and by the
real engine (Phase 1).

Owner B's Gymnasium wrapper and owner C's console/harness import *this*, never a
concrete class, so the real engine drops in behind an unchanged interface.

Plan reference: section 6.1 ("break the serial dependency on day one").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import MatchResult, MatchState, SegmentResult, Side, Tactic


class PitchSim(ABC):
    """A single football match, advanced between coaching decision epochs.

    Contract:
      * `reset` must be callable repeatedly; a given seed reproduces a match exactly.
      * `advance` simulates forward and returns at the *next* decision epoch,
        which may arrive early on a trigger (goal, red card, half-time). The
        returned `duration_minutes` is the SMDP dwell time tau_k.
      * `set_tactic` takes effect from the next call to `advance`.
      * `replay` is only guaranteed to be populated when `record=True`.
    """

    #: Bumped when the observable-statistics contract changes.
    STATS_VERSION = 1

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        home_tactic: Tactic | None = None,
        away_tactic: Tactic | None = None,
        record: bool = False,
    ) -> MatchState:
        """Start a new match and return the state at kickoff."""

    @abstractmethod
    def set_tactic(self, side: Side, tactic: Tactic) -> None:
        """Apply a coaching decision for `side`, effective from the next advance."""

    @abstractmethod
    def advance(self, max_minutes: float) -> SegmentResult:
        """Simulate forward by at most `max_minutes`, stopping early on a trigger."""

    @property
    @abstractmethod
    def state(self) -> MatchState:
        """Current match state."""

    @property
    def done(self) -> bool:
        return self.state.done

    @abstractmethod
    def result(self) -> MatchResult:
        """Terminal summary. Only valid once `done` is True."""

    def replay(self) -> dict[str, Any]:
        """Replay JSON for the console. See `pitchsim.replay.ReplayBuilder`.

        Implementations that were not constructed with `record=True` may return a
        replay with an empty `ticks` list; the schema is still valid.
        """
        raise NotImplementedError(f"{type(self).__name__} does not record replays")

"""PITCHSIM — the GAFFER-RL football simulator.

Phase 0 ships the interface (`base`, `types`, `replay`) plus a stub engine, so
that the learning and console workstreams are never blocked on the physics.
Phase 1 adds `engine`, `controller`, `pitch_control` and `calibrate` behind the
same interface.
"""

from .base import PitchSim
from .replay import REPLAY_SCHEMA_VERSION, ReplayBuilder, load_replay, validate_replay
from .stub import StubPitchSim
from .types import (
    BLOCK_HEIGHTS,
    FORMATIONS,
    HEAD_SIZES,
    MASK_LENGTH,
    N_ARCHETYPES,
    PRESS_LEVELS,
    SCHEMA_VERSION,
    SUBSTITUTIONS,
    TEMPOS,
    MatchResult,
    MatchState,
    SegmentResult,
    Side,
    SquadState,
    Tactic,
    WindowStats,
)

__all__ = [
    "PitchSim", "StubPitchSim",
    "Tactic", "Side", "MatchState", "MatchResult", "SegmentResult",
    "SquadState", "WindowStats",
    "FORMATIONS", "BLOCK_HEIGHTS", "PRESS_LEVELS", "TEMPOS", "SUBSTITUTIONS",
    "HEAD_SIZES", "MASK_LENGTH", "N_ARCHETYPES", "SCHEMA_VERSION",
    "ReplayBuilder", "REPLAY_SCHEMA_VERSION", "load_replay", "validate_replay",
]

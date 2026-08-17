"""Scripted opponent coaches — the K archetypes the belief filter must identify.

Two things here are load-bearing for the whole project:

1. **Archetypes must be statistically distinguishable** from their observable
   footprint alone, or Pillar 1 has nothing to infer (Gate 1, design note 3.2).
2. **They must change plan mid-match.** A coach who picks a shape at kickoff and
   holds it for ninety minutes lets the belief filter saturate by minute ten,
   collapsing `belief ~= oracle` and flattening the headline ablation. Switching
   is what keeps the filtering problem alive and makes adaptation latency — time
   to *re*-identify after a true switch — a meaningful metric (design note 3.4).

Batched over matches to match `pitchsim.engine.BatchedEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pitchsim.controller import load_config
from pitchsim.types import (
    BLOCK_HEIGHTS,
    FORMATIONS,
    PRESS_LEVELS,
    TEMPOS,
    Tactic,
)


@lru_cache(maxsize=1)
def archetype_table() -> tuple[list[str], list[str], np.ndarray]:
    """(keys, labels, base tactics (K, 5))."""
    tactics_cfg, _ = load_config()
    entries = tactics_cfg["archetypes"]
    keys = list(entries)
    labels = [entries[k]["label"] for k in keys]

    base = np.zeros((len(keys), 5), dtype=np.int64)
    for i, key in enumerate(keys):
        spec = entries[key]["base"]
        base[i] = Tactic.from_names(
            formation=spec["formation"],
            block_height=spec["block_height"],
            press=spec["press"],
            tempo=spec["tempo"],
        ).as_array()
    return keys, labels, base


@lru_cache(maxsize=1)
def switching_rules() -> tuple[float, list[dict]]:
    tactics_cfg, _ = load_config()
    switching = tactics_cfg["switching"]
    return float(switching["hazard_per_epoch"]), list(switching["rules"])


N_ARCHETYPES = len(archetype_table()[0])

_HEAD_LOOKUP = {
    "formation": (0, FORMATIONS),
    "block_height": (1, BLOCK_HEIGHTS),
    "press": (2, PRESS_LEVELS),
    "tempo": (3, TEMPOS),
}


@dataclass
class ScriptedCoaches:
    """A batch of scripted coaches, one per match.

    `assignment` is (B,) archetype indices. Call `decide` at each decision epoch.
    """

    assignment: np.ndarray

    def __post_init__(self) -> None:
        self.assignment = np.asarray(self.assignment, dtype=np.int64)
        self._switched = np.zeros(self.assignment.shape, dtype=bool)
        #: Epoch index at which each match's coach last changed plan. The belief
        #: filter's adaptation latency is measured against exactly this.
        self.last_switch_epoch = np.zeros(self.assignment.shape, dtype=np.int64)
        self._epoch = 0

    def reset(self) -> None:
        self._switched[:] = False
        self.last_switch_epoch[:] = 0
        self._epoch = 0

    def decide(
        self,
        minute: np.ndarray,
        score_diff: np.ndarray,
        rng: np.random.Generator,
        red_card: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (B, 5) tactic head indices for this epoch.

        `score_diff` is from this coach's own perspective.
        """
        self._epoch += 1
        _, _, base = archetype_table()
        tactics = base[self.assignment].copy()

        hazard, rules = switching_rules()
        changed = np.zeros(len(tactics), dtype=bool)

        for rule in rules:
            when = rule["when"]
            mask = np.ones(len(tactics), dtype=bool)

            if "trailing_by_at_least" in when:
                mask &= score_diff <= -int(when["trailing_by_at_least"])
            if "leading_by_at_least" in when:
                mask &= score_diff >= int(when["leading_by_at_least"])
            if "after_minute" in when:
                mask &= minute >= float(when["after_minute"])
            if when.get("red_card_against_self") and red_card is not None:
                mask &= red_card.astype(bool)
            elif when.get("red_card_against_self"):
                mask &= False

            if not mask.any():
                continue
            for head, value in rule["shift"].items():
                index, names = _HEAD_LOOKUP[head]
                tactics[mask, index] = names.index(value)
            changed |= mask

        # Random drift on top of the state-dependent rules, so the filter cannot
        # simply learn the rule table and predict switches deterministically.
        drift = rng.random(len(tactics)) < hazard
        if drift.any():
            for index, names in (
                (1, BLOCK_HEIGHTS), (2, PRESS_LEVELS), (3, TEMPOS)
            ):
                pick = rng.integers(0, len(names), size=len(tactics))
                tactics[drift, index] = pick[drift]
            changed |= drift

        newly = changed & ~self._switched
        self.last_switch_epoch[newly] = self._epoch
        self._switched |= changed
        return tactics


def sample_assignment(batch: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random archetype per match — the Stage-1 gauntlet opponent."""
    return rng.integers(0, N_ARCHETYPES, size=batch)

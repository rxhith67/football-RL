"""Persistent checkpoint pool for prioritised fictitious self-play (PFSP)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "checkpoints" / "league.json"


@dataclass
class LeagueEntry:
    checkpoint: str
    rating: float = 1500.0
    games: int = 0
    wins_against_current: float = 0.5


@dataclass
class PFSPLeague:
    """Small, serialisable PFSP pool with Elo bookkeeping."""

    entries: list[LeagueEntry] = field(default_factory=list)
    path: Path = DEFAULT_MANIFEST

    @classmethod
    def load(cls, path: Path = DEFAULT_MANIFEST) -> "PFSPLeague":
        if not path.exists():
            return cls(path=path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls([LeagueEntry(**row) for row in blob.get("entries", [])], path)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "entries": [entry.__dict__ for entry in self.entries],
        }, indent=2), encoding="utf-8")
        return self.path

    def register(self, checkpoint: str | Path, rating: float = 1500.0) -> None:
        value = str(Path(checkpoint).resolve())
        if not any(entry.checkpoint == value for entry in self.entries):
            self.entries.append(LeagueEntry(value, rating=rating))
            self.save()

    def sample(self, rng: np.random.Generator) -> Path:
        if not self.entries:
            raise RuntimeError(
                "PFSP league is empty; train Stage 1 first so it can seed the pool"
            )
        # Prioritise opponents the current policy has not mastered.
        weights = np.array([max(1.0 - e.wins_against_current, 0.05)
                            for e in self.entries], dtype=np.float64)
        weights /= weights.sum()
        return Path(self.entries[int(rng.choice(len(self.entries), p=weights))].checkpoint)

    def record(self, checkpoint: str | Path, score: float, k: float = 32.0) -> None:
        """Update an opponent after a score in [0, 1] by the current policy."""
        value = str(Path(checkpoint).resolve())
        entry = next((e for e in self.entries if e.checkpoint == value), None)
        if entry is None:
            raise KeyError(value)
        entry.games += 1
        entry.wins_against_current += (float(score) - entry.wins_against_current) / entry.games
        expected = 1.0 / (1.0 + 10.0 ** ((entry.rating - 1500.0) / 400.0))
        entry.rating -= k * (float(score) - expected)
        self.save()

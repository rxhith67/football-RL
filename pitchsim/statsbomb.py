"""StatsBomb open-data loader — shots for the xG fit and match aggregates for
calibration.

Data: https://github.com/statsbomb/open-data (free, but see their terms — if this
repository is public, the StatsBomb attribution requirement applies and is
carried in the README).

Everything is cached under `data/statsbomb/raw/` so a fit is reproducible offline
and the demo never depends on the network (plan section 6.2, Gate 4).

CLI::

    python -m pitchsim.statsbomb --fetch --max-matches 200
    python -m pitchsim.statsbomb --summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "statsbomb" / "raw"
SHOTS_PARQUET = ROOT / "data" / "statsbomb" / "shots.parquet"

#: Competitions with full event coverage and a senior top-flight profile, which is
#: what the calibration table in the plan targets. The xG fit wants several
#: thousand shots; a single season is not enough for a stable held-out estimate.
DEFAULT_COMPETITIONS: tuple[tuple[int, int], ...] = (
    (11, 90),    # La Liga 2020/2021
    (11, 42),    # La Liga 2019/2020
    (2, 44),     # Premier League 2003/2004
    (9, 281),    # 1. Bundesliga 2023/2024
    (55, 43),    # UEFA Euro 2020
    (43, 106),   # FIFA World Cup 2022
    (72, 107),   # Women's World Cup 2023
    (53, 106),   # UEFA Women's Euro 2022
)

#: Event JSONs are ~2-4 MB each and are needed only long enough to pull the shots
#: out. Caching 200 of them costs ~600 MB for no benefit once `shots.parquet`
#: exists, so streaming is the default and caching is opt-in.
CACHE_EVENTS = False

# StatsBomb pitch is 120 x 80 yards-ish units; goal centre at (120, 40).
SB_GOAL_X = 120.0
SB_GOAL_Y = 40.0
SB_GOAL_HALF_WIDTH = 3.66      # metres, converted below
SB_X_TO_M = 105.0 / 120.0
SB_Y_TO_M = 68.0 / 80.0


def _get_json(url: str, cache_path: Path | None, pause: float = 0.10) -> Any:
    """Fetch, optionally via an on-disk cache. Returns None if unavailable.

    `cache_path=None` streams: the JSON is parsed and discarded, never written.
    """
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache_path.unlink()   # corrupt cache entry; re-fetch

    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"  ! network error for {url}: {exc}")
        return None
    if response.status_code != 200:
        print(f"  ! HTTP {response.status_code} for {url}")
        return None

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(response.text, encoding="utf-8")
        except OSError as exc:
            # Out of disk is not fatal — we only needed the parsed object.
            print(f"  ! cache write skipped ({exc.strerror})")
    time.sleep(pause)             # be polite to the CDN
    return response.json()


def fetch_matches(competition_id: int, season_id: int) -> list[dict]:
    data = _get_json(
        f"{BASE}/matches/{competition_id}/{season_id}.json",
        CACHE / "matches" / f"{competition_id}_{season_id}.json",
    )
    return data or []


def fetch_events(match_id: int, cache: bool = CACHE_EVENTS) -> list[dict]:
    path = CACHE / "events" / f"{match_id}.json" if cache else None
    data = _get_json(f"{BASE}/events/{match_id}.json", path)
    return data or []


# --------------------------------------------------------------------------
# Shot extraction
# --------------------------------------------------------------------------


def _shot_geometry(x: float, y: float) -> tuple[float, float]:
    """Distance (m) and visible goal angle (rad) from a StatsBomb location."""
    dx = (SB_GOAL_X - x) * SB_X_TO_M
    dy = (y - SB_GOAL_Y) * SB_Y_TO_M
    distance = float(np.hypot(dx, dy))

    # Angle subtended by the goal mouth at the shot location.
    a = np.hypot(dx, dy - SB_GOAL_HALF_WIDTH)
    b = np.hypot(dx, dy + SB_GOAL_HALF_WIDTH)
    cos_theta = (a**2 + b**2 - (2 * SB_GOAL_HALF_WIDTH) ** 2) / (2 * a * b + 1e-9)
    angle = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return distance, angle


def _freeze_frame_features(shot: dict, x: float, y: float) -> dict[str, float]:
    """Geometry of the defenders and goalkeeper at the moment of the shot.

    Goalkeeper positioning is the single largest signal in the freeze frame after
    distance and angle: a keeper caught off his line, or displaced laterally from
    the shot-to-goal line, changes conversion far more than an extra outfield
    body in the cone.
    """
    frame = shot.get("freeze_frame") or []
    out = {
        "defenders_in_cone": 0.0,
        "nearest_defender_m": 15.0,   # neutral default when the frame is absent
        "gk_off_line_m": 1.5,
        "gk_lateral_m": 0.0,
        "has_freeze_frame": 0.0,
    }
    if not frame:
        return out
    out["has_freeze_frame"] = 1.0

    posts = [(SB_GOAL_X, SB_GOAL_Y - 4.0), (SB_GOAL_X, SB_GOAL_Y + 4.0)]

    def side(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    nearest = np.inf
    count = 0
    for actor in frame:
        loc = actor.get("location")
        if not loc:
            continue
        px, py = float(loc[0]), float(loc[1])
        is_keeper = actor.get("position", {}).get("name") == "Goalkeeper"
        teammate = actor.get("teammate", True)

        if is_keeper and not teammate:
            out["gk_off_line_m"] = abs(SB_GOAL_X - px) * SB_X_TO_M
            # Perpendicular displacement from the shooter-to-goal-centre line.
            dx, dy = SB_GOAL_X - x, SB_GOAL_Y - y
            norm = np.hypot(dx, dy) + 1e-9
            out["gk_lateral_m"] = abs((px - x) * dy - (py - y) * dx) / norm * SB_Y_TO_M
            continue

        if teammate:
            continue

        dist = np.hypot((px - x) * SB_X_TO_M, (py - y) * SB_Y_TO_M)
        nearest = min(nearest, dist)

        if px <= x:            # behind the shooter, cannot block
            continue
        s1 = side(x, y, *posts[0], px, py)
        s2 = side(*posts[0], *posts[1], px, py)
        s3 = side(*posts[1], x, y, px, py)
        if (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0):
            count += 1

    out["defenders_in_cone"] = float(count)
    if np.isfinite(nearest):
        out["nearest_defender_m"] = float(min(nearest, 30.0))
    return out


def extract_shots(events: Iterable[dict], match_id: int) -> list[dict]:
    rows: list[dict] = []
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        shot = event.get("shot", {})
        loc = event.get("location")
        if not loc or len(loc) < 2:
            continue

        # Penalties are a separate ~0.76 lottery and would distort the fit.
        shot_type = shot.get("type", {}).get("name", "")
        if shot_type == "Penalty":
            continue

        x, y = float(loc[0]), float(loc[1])
        distance, angle = _shot_geometry(x, y)
        body = shot.get("body_part", {}).get("name", "")
        outcome = shot.get("outcome", {}).get("name", "")
        technique = shot.get("technique", {}).get("name", "")
        pattern = event.get("play_pattern", {}).get("name", "")

        row = {
            "match_id": match_id,
            "x": x,
            "y": y,
            "distance_m": distance,
            "angle_rad": angle,
            "is_header": int(body == "Head"),
            "is_open_play": int(shot_type == "Open Play"),
            "under_pressure": int(bool(event.get("under_pressure", False))),
            "is_first_time": int(bool(shot.get("first_time", False))),
            "is_one_on_one": int(bool(shot.get("one_on_one", False))),
            "is_volley": int(technique == "Volley"),
            "from_counter": int(pattern == "From Counter"),
            "goal": int(outcome == "Goal"),
            "sb_xg": shot.get("statsbomb_xg", np.nan),
        }
        row.update(_freeze_frame_features(shot, x, y))
        rows.append(row)
    return rows


def build_shot_table(
    competitions: tuple[tuple[int, int], ...] = DEFAULT_COMPETITIONS,
    max_matches: int = 200,
    save: bool = True,
) -> pd.DataFrame:
    """Download (or read from cache) and assemble the shot table."""
    per_competition: list[list[int]] = []
    for comp_id, season_id in competitions:
        matches = fetch_matches(comp_id, season_id)
        print(f"competition {comp_id}/{season_id}: {len(matches)} matches")
        per_competition.append([int(m["match_id"]) for m in matches])

    # Interleave rather than concatenate, so a `max_matches` cut leaves a mix of
    # competitions instead of exhausting the first one.
    match_ids: list[int] = []
    for i in range(max(len(c) for c in per_competition) if per_competition else 0):
        for competition in per_competition:
            if i < len(competition):
                match_ids.append(competition[i])
    match_ids = match_ids[:max_matches]
    print(f"extracting shots from {len(match_ids)} matches...")

    rows: list[dict] = []
    for i, match_id in enumerate(match_ids, 1):
        events = fetch_events(match_id)
        if events:
            rows.extend(extract_shots(events, match_id))
        if i % 25 == 0:
            print(f"  {i}/{len(match_ids)} matches, {len(rows)} shots")

    df = pd.DataFrame(rows)
    if save and not df.empty:
        SHOTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SHOTS_PARQUET, index=False)
        print(f"saved {len(df)} shots -> {SHOTS_PARQUET}")
    return df


def load_shots() -> pd.DataFrame:
    if not SHOTS_PARQUET.exists():
        raise FileNotFoundError(
            f"{SHOTS_PARQUET} not found. Run: python -m pitchsim.statsbomb --fetch"
        )
    return pd.read_parquet(SHOTS_PARQUET)


def calibration_summary(df: pd.DataFrame) -> dict[str, float]:
    """Real-world reference values for the plan's calibration table."""
    per_match = df.groupby("match_id").agg(
        shots=("goal", "size"), goals=("goal", "sum"), xg=("sb_xg", "sum")
    )
    return {
        "n_matches": int(len(per_match)),
        "n_shots": int(len(df)),
        "goals_per_match": float(per_match["goals"].mean()),
        "shots_per_team": float(per_match["shots"].mean() / 2.0),
        "xg_per_shot": float(df["sb_xg"].mean()),
        "conversion_rate": float(df["goal"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="StatsBomb open-data loader")
    parser.add_argument("--fetch", action="store_true", help="download and build shots.parquet")
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument("--summary", action="store_true", help="print calibration reference values")
    args = parser.parse_args()

    if args.fetch:
        build_shot_table(max_matches=args.max_matches)
    if args.summary or not args.fetch:
        summary = calibration_summary(load_shots())
        print("\nReal-world calibration reference (StatsBomb open data)")
        print("-" * 55)
        for key, value in summary.items():
            print(f"  {key:22} {value:.4f}" if isinstance(value, float) else f"  {key:22} {value}")


if __name__ == "__main__":
    main()

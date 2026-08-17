"""Expected possession value — the "how good is the ball here?" surface.

The event layer scores every on-ball option as `pitch_control(target) x EPV(target)`
minus a risk term (design note 3.2), so EPV is what makes a switch to a free
winger worth more than a square pass to a covered centre-back.

This is a *proxy*, and the write-up must say so. A properly fitted expected-threat
model needs possession-chain data, which the shot table does not carry. Instead
the grid blends two quantities that are grounded:

  * the fitted xG of a shot taken from that cell (real, from `pitchsim.xg`), and
  * a progression term rewarding territory gained toward the opponent's goal.

The blend is smoothed so that the resulting surface has no cliff edges for the
softmax to exploit. An addition to the plan's section 8 skeleton, kept separate
from `pitch_control` because value and control are different quantities that
happen to be multiplied together.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .pitch_control import shot_geometry

PITCH_L, PITCH_W = 105.0, 68.0

#: EPV is expressed on the SAME SCALE AS xG — the probability that this possession
#: ends in a goal — so the event layer compares "shoot now" against "move the ball
#: somewhere better" without an arbitrary conversion factor.
#:
#: This ceiling MUST exceed the best available xG (~0.35 from six yards). Having
#: the ball under control in the six-yard box is worth strictly more than the
#: single shot you could take from there, because you retain the option to take a
#: better one. Setting it below max-xG makes shooting weakly dominate passing
#: everywhere in the final third, which showed up as 100 shots per team per match.
EPV_MAX = 0.45


@lru_cache(maxsize=16)
def epv_grid(nx: int = 105, ny: int = 68, shot_weight: float = 0.62,
             epv_max: float = EPV_MAX) -> np.ndarray:
    """Build the EPV lookup for a team attacking +x. Shape (nx, ny), values in [0, 1].

    Cached: this is built once and indexed thereafter, never recomputed in the loop.
    """
    from .xg import XGModel

    xs = (np.arange(nx) + 0.5) * PITCH_L / nx
    ys = (np.arange(ny) + 0.5) * PITCH_W / ny
    mesh_x, mesh_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([mesh_x, mesh_y], axis=-1)

    distance, angle = shot_geometry(points, goal_x=PITCH_L)

    try:
        model = XGModel.load()
        flat_d, flat_a = distance.ravel(), angle.ravel()
        rows = np.tile(
            np.array([[model.DEFAULTS[name] for name in model.feature_names]]),
            (flat_d.size, 1),
        )
        names = list(model.feature_names)
        rows[:, names.index("distance_m")] = flat_d
        rows[:, names.index("angle_rad")] = flat_a
        shot_value = model.predict(rows).reshape(distance.shape)
    except FileNotFoundError:
        # Analytic fallback so the engine still runs before the xG fit exists.
        shot_value = np.exp(-distance / 9.0) * (angle / np.pi) ** 0.5

    # Territory: value rises toward the opponent's goal, mildly toward the middle.
    progression = (mesh_x / PITCH_L) ** 2.2
    centrality = 1.0 - 0.35 * (np.abs(mesh_y - PITCH_W / 2.0) / (PITCH_W / 2.0)) ** 2

    value = shot_weight * shot_value + (1.0 - shot_weight) * progression * centrality
    value = _smooth(value)
    return (epv_max * value / value.max()).astype(np.float32)


def _smooth(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Cheap separable box blur — removes cliffs the option softmax could exploit."""
    out = grid.astype(np.float64)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:] + 2.0 * out) / 6.0
    return out


def lookup(points: np.ndarray, attacking_positive_x: np.ndarray,
           epv_max: float = EPV_MAX) -> np.ndarray:
    """EPV at arbitrary points, mirrored for the side attacking -x.

    Parameters
    ----------
    points : (..., 2) pitch coordinates in metres
    attacking_positive_x : (...,) bool — True for home, False for away

    Returns
    -------
    (...,) EPV in [0, 1].
    """
    grid = epv_grid(epv_max=epv_max)
    nx, ny = grid.shape

    x = np.where(attacking_positive_x, points[..., 0], PITCH_L - points[..., 0])
    y = np.where(attacking_positive_x, points[..., 1], PITCH_W - points[..., 1])

    ix = np.clip((x / PITCH_L * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((y / PITCH_W * ny).astype(np.int64), 0, ny - 1)
    return grid[ix, iy]

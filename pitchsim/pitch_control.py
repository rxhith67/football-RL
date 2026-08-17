"""Pitch control — a simplified Spearman-style time-to-intercept surface.

Every array operation here is batched over matches AND over candidate target
points, because this is the inner loop of the event layer: it is evaluated once
per on-ball decision, for ten candidate receivers, for every match in the batch.
A per-match Python loop at this level would miss the 300 matches/min target on
its own.

Model
-----
A player reaches a point at

    t = reaction_time + |target - (position + velocity * reaction_time)| / max_speed

and the probability that team A arrives first is a logistic function of the gap
between the two teams' fastest arrivals:

    P(A first) = sigmoid((t_Bmin - t_Amin) / sigma)

This is the tractable core of Spearman's model (MIT SSAC, "Beyond Expected
Goals"): it keeps time-to-intercept and a soft, calibratable contest, and drops
the full probabilistic-arrival integral, which contributes little at 5 Hz and
costs an order of magnitude more compute. The Friends-of-Tracking
`Metrica_PitchControl.py` reference implementation is the pattern; this is a
re-implementation in vectorised form, not a copy.

Plan reference: section 3.2.
"""

from __future__ import annotations

import numpy as np


def time_to_intercept(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    max_speed: np.ndarray,
    reaction_time: float = 0.7,
) -> np.ndarray:
    """Time for each player to reach each target.

    Parameters
    ----------
    positions : (..., N, 2)
    velocities : (..., N, 2)
    targets : (..., M, 2)
    max_speed : (..., N) or scalar

    Returns
    -------
    (..., N, M) arrival times in seconds.
    """
    # Where a player can get to "for free" during the reaction window.
    start = positions + velocities * reaction_time          # (..., N, 2)
    delta = targets[..., None, :, :] - start[..., :, None, :]  # (..., N, M, 2)
    distance = np.sqrt(np.einsum("...d,...d->...", delta, delta))
    speed = np.asarray(max_speed)
    if speed.ndim:
        speed = speed[..., :, None]
    return reaction_time + distance / np.maximum(speed, 1e-6)


def control_probability(
    own_positions: np.ndarray,
    own_velocities: np.ndarray,
    opp_positions: np.ndarray,
    opp_velocities: np.ndarray,
    targets: np.ndarray,
    own_max_speed: np.ndarray,
    opp_max_speed: np.ndarray,
    reaction_time: float = 0.7,
    sigma: float = 0.45,
    exclude_own: int | None = None,
) -> np.ndarray:
    """Probability that the OWN team controls each target point.

    `exclude_own` drops one own-team index from the contest — used so that the
    player making a pass is not counted as a candidate receiver of his own ball.

    Returns (..., M) in [0, 1].
    """
    t_own = time_to_intercept(own_positions, own_velocities, targets,
                              own_max_speed, reaction_time)
    if exclude_own is not None:
        t_own = t_own.copy()
        t_own[..., exclude_own, :] = np.inf

    t_opp = time_to_intercept(opp_positions, opp_velocities, targets,
                              opp_max_speed, reaction_time)

    best_own = t_own.min(axis=-2)
    best_opp = t_opp.min(axis=-2)
    return 1.0 / (1.0 + np.exp(-(best_opp - best_own) / max(sigma, 1e-6)))


def pass_success_probability(
    origin: np.ndarray,
    targets: np.ndarray,
    opponents: np.ndarray,
    opponent_velocities: np.ndarray,
    opponent_max_speed: np.ndarray,
    destination_control: np.ndarray,
    pass_speed: float = 26.0,
    reaction_time: float = 0.7,
    sigma: float = 0.45,
) -> np.ndarray:
    """Probability a pass arrives, accounting for interception ALONG THE LANE.

    Destination control alone is not a passing model. Evaluating control at a
    teammate's own feet trivially favours that teammate — he is already standing
    there — so every ball into the box completes and the sim produces a hundred
    shots a game. What actually kills a pass in football is a defender stepping
    into the lane before the ball gets past him.

    For each opponent we find the closest point on the pass segment, compare the
    time the ball needs to reach that point against the time the defender needs,
    and combine the per-defender interception probabilities independently.

    Parameters
    ----------
    origin : (m, 2)
    targets : (m, K, 2)
    opponents : (m, N, 2)
    destination_control : (m, K) — from `control_probability`

    Returns
    -------
    (m, K) success probabilities.
    """
    segment = targets - origin[:, None, :]                       # (m, K, 2)
    length = np.linalg.norm(segment, axis=-1)                    # (m, K)
    unit = segment / np.maximum(length, 1e-6)[..., None]

    # Project each opponent onto each pass lane.
    relative = opponents[:, None, :, :] - origin[:, None, None, :]   # (m, 1, N, 2)
    along = np.einsum("mkd,mknd->mkn", unit, np.broadcast_to(
        relative, (*segment.shape[:2], opponents.shape[1], 2)))
    along = np.clip(along, 0.0, length[..., None])

    closest = origin[:, None, None, :] + unit[:, :, None, :] * along[..., None]
    to_lane = np.linalg.norm(opponents[:, None, :, :] - closest, axis=-1)  # (m, K, N)

    time_ball = along / max(pass_speed, 1e-6)
    speed = opponent_max_speed[:, None, :]
    time_defender = reaction_time + to_lane / np.maximum(speed, 1e-6)

    # Sigmoid rather than a hard test: a defender who arrives marginally early
    # does not always win the ball.
    p_intercept = 1.0 / (1.0 + np.exp(-(time_ball - time_defender) / max(sigma, 1e-6)))

    # The keeper is index 0 and should not be treated as a lane interceptor.
    p_intercept[:, :, 0] = 0.0

    p_clear = np.prod(1.0 - p_intercept, axis=-1)                # (m, K)

    # Both factors matter and neither gets a floor: the ball must survive the
    # lane AND the receiver must actually win it at the other end. An earlier
    # version floored the destination term at 0.55, which let a tightly marked
    # striker in the six-yard box collect the ball more than half the time and
    # was a large part of why teams generated ten xG a match.
    return p_clear * destination_control


def control_surface(
    own_positions: np.ndarray,
    own_velocities: np.ndarray,
    opp_positions: np.ndarray,
    opp_velocities: np.ndarray,
    own_max_speed: np.ndarray,
    opp_max_speed: np.ndarray,
    grid: tuple[int, int] = (52, 34),
    pitch: tuple[float, float] = (105.0, 68.0),
    reaction_time: float = 0.7,
    sigma: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full control heatmap for ONE match — the console's toggleable overlay.

    Deliberately not called from the simulation loop: the event layer needs
    control at ten candidate points, not at 1,768 grid cells. This exists for
    rendering and for the calibration figures.

    Returns (surface (ny, nx), xs, ys).
    """
    nx, ny = grid
    xs = np.linspace(0.0, pitch[0], nx)
    ys = np.linspace(0.0, pitch[1], ny)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    targets = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)

    probability = control_probability(
        own_positions, own_velocities, opp_positions, opp_velocities,
        targets, own_max_speed, opp_max_speed, reaction_time, sigma,
    )
    return probability.reshape(ny, nx), xs, ys


def defenders_in_cone(
    shooter: np.ndarray,
    opponents: np.ndarray,
    goal_x: float,
    goal_y: float = 34.0,
    half_width: float = 3.66,
) -> np.ndarray:
    """Count opponents inside the shooter -> goalposts triangle.

    Matches the feature of the same name that `pitchsim.statsbomb` extracts from
    real freeze frames, so the xG model is fed the quantity it was fitted on.

    Parameters
    ----------
    shooter : (..., 2)
    opponents : (..., N, 2)

    Returns
    -------
    (...,) integer counts.
    """
    shape = shooter.shape[:-1]
    gx = np.broadcast_to(np.asarray(goal_x, dtype=np.float64), shape)
    post_a = np.stack([gx, np.broadcast_to(np.float64(goal_y - half_width), shape)], axis=-1)
    post_b = np.stack([gx, np.broadcast_to(np.float64(goal_y + half_width), shape)], axis=-1)

    def cross(a, b, p):
        return ((b[..., None, 0] - a[..., None, 0]) * (p[..., 1] - a[..., None, 1])
                - (b[..., None, 1] - a[..., None, 1]) * (p[..., 0] - a[..., None, 0]))

    s1 = cross(shooter, post_a, opponents)
    s2 = cross(post_a, post_b, opponents)
    s3 = cross(post_b, shooter, opponents)

    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))

    # A defender behind the shooter (relative to the goal) cannot block.
    toward_goal = np.sign(goal_x - shooter[..., 0])[..., None]
    ahead = (opponents[..., 0] - shooter[..., None, 0]) * toward_goal > 0
    return np.sum(inside & ahead, axis=-1)


def shot_geometry(
    positions: np.ndarray, goal_x: float, goal_y: float = 34.0, half_width: float = 3.66
) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) and visible goal angle (rad), matching the xG model's features."""
    dx = goal_x - positions[..., 0]
    dy = positions[..., 1] - goal_y
    distance = np.hypot(dx, dy)

    a = np.hypot(dx, dy - half_width)
    b = np.hypot(dx, dy + half_width)
    cos_theta = (a**2 + b**2 - (2 * half_width) ** 2) / (2 * a * b + 1e-9)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return distance, angle

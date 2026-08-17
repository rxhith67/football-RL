"""The parameterised low-level player controller — a FIXED intra-option policy.

This layer is deliberately hand-written and never learned. In the language of
Sutton, Precup and Singh (1999), the coach learns a policy *over options* whose
intra-option policies are given. Saying so converts "we didn't train the players"
from a weakness into a design decision with a citation, and it is written into
the spec precisely so that scope creep into player-level learning can be refused
(design note 3.1 and 7).

Everything is batched over matches: arrays are shaped (B, 2, 11, ...) for
[match, side, player]. Home (index 0) attacks +x; away attacks -x.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
PITCH_L, PITCH_W = 105.0, 68.0
HOME, AWAY = 0, 1


@lru_cache(maxsize=1)
def load_config() -> tuple[dict, dict]:
    """(tactics.yaml, pitchsim.yaml) — cached, since this is read per reset."""
    tactics = yaml.safe_load((ROOT / "config" / "tactics.yaml").read_text(encoding="utf-8"))
    physics = yaml.safe_load((ROOT / "config" / "pitchsim.yaml").read_text(encoding="utf-8"))
    return tactics, physics


@lru_cache(maxsize=1)
def formation_tables() -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """(slots (5, 11, 2) normalised, widths (5,), centre-back indices per formation)."""
    tactics, _ = load_config()
    names = tactics["action_space"]["heads"]["formation"]
    slots, widths, centre_backs = [], [], []
    for name in names:
        entry = tactics["formations"][name]
        slots.append(entry["slots"])
        widths.append(entry["width"])
        centre_backs.append(entry["centre_backs"])
    return (np.array(slots, dtype=np.float32),
            np.array(widths, dtype=np.float32),
            centre_backs)


@lru_cache(maxsize=1)
def modifier_tables() -> dict[str, np.ndarray]:
    """Head modifiers as arrays indexed by the action head value."""
    tactics, _ = load_config()
    mods = tactics["modifiers"]

    def column(head: str, key: str) -> np.ndarray:
        names = tactics["action_space"]["heads"][head]
        return np.array([mods[head][n][key] for n in names], dtype=np.float32)

    return {
        "line_x": column("block_height", "line_x"),
        "compactness": column("block_height", "compactness"),
        "offside_trap": column("block_height", "offside_trap"),
        "trigger_radius": column("press", "trigger_radius"),
        "closing_speed": column("press", "closing_speed"),
        "stamina_drain": column("press", "stamina_drain"),
        "foul_rate": column("press", "foul_rate"),
        "softmax_temp": column("tempo", "softmax_temp"),
        "risk_weight": column("tempo", "risk_weight"),
        "directness": column("tempo", "directness"),
    }


def base_slots(formation_idx: np.ndarray) -> np.ndarray:
    """Formation slots in metres, oriented per side.

    Parameters
    ----------
    formation_idx : (B, 2) int

    Returns
    -------
    (B, 2, 11, 2) float32 — absolute pitch coordinates.
    """
    slots, _, _ = formation_tables()
    normalised = slots[formation_idx]                     # (B, 2, 11, 2)
    out = np.empty_like(normalised)

    # Home: x grows from its own goal at x=0 toward x=105.
    out[:, HOME, :, 0] = normalised[:, HOME, :, 0] * PITCH_L
    out[:, HOME, :, 1] = normalised[:, HOME, :, 1] * PITCH_W
    # Away: mirrored — its own goal is at x=105.
    out[:, AWAY, :, 0] = PITCH_L - normalised[:, AWAY, :, 0] * PITCH_L
    out[:, AWAY, :, 1] = PITCH_W - normalised[:, AWAY, :, 1] * PITCH_W
    return out


def target_points(
    slots: np.ndarray,
    ball: np.ndarray,
    possessing_side: np.ndarray,
    formation_idx: np.ndarray,
    block_idx: np.ndarray,
    press_idx: np.ndarray,
    tempo_idx: np.ndarray,
    shape_follow: float = 0.55,
) -> np.ndarray:
    """Where each player wants to be this tick.

    Three effects compose, in order of magnitude:

    1. **Ball translation.** The whole shape slides toward the ball, scaled by
       the block's compactness. This is what makes a low block visibly compact
       and a high line visibly stretched.
    2. **Block height.** Out of possession, the defensive line seeks its rest
       position (`line_x`); in possession the shape pushes up instead.
    3. **Width.** Formation width scales lateral spread about the centre, and
       direct tempo narrows the shape slightly as players attack the box.

    Returns (B, 2, 11, 2).
    """
    mods = modifier_tables()
    _, widths, _ = formation_tables()

    compact = mods["compactness"][block_idx][..., None, None]      # (B, 2, 1, 1)
    line_x = mods["line_x"][block_idx]                             # (B, 2)
    directness = mods["directness"][tempo_idx][..., None]          # (B, 2, 1)
    width = widths[formation_idx][..., None]                       # (B, 2, 1)

    ball_xy = ball[:, None, None, :]                               # (B, 1, 1, 2)
    centre = np.array([PITCH_L / 2.0, PITCH_W / 2.0], dtype=np.float32)

    # 1. slide toward the ball
    # `shape_follow` caps how far the whole shape slides toward the ball. This is
    # the single largest driver of how much a team can camp in the final third,
    # and therefore of how much stronger an aggressive shape is than a balanced
    # one. At 0.55 extreme tactics generated ~3.5x the chances of balanced ones no
    # matter how the shooting and tackling knobs were set, which is why the
    # tactic-space calibration could never be closed by parameter search.
    targets = slots + compact * shape_follow * (ball_xy - centre)

    # 2. block height: out of possession, seek the line; in possession, push up
    in_possession = (possessing_side[:, None] == np.array([HOME, AWAY])[None, :])
    own_goal_x = np.array([0.0, PITCH_L], dtype=np.float32)[None, :]
    attack_sign = np.array([1.0, -1.0], dtype=np.float32)[None, :]

    rest_x = own_goal_x + attack_sign * line_x                     # (B, 2)
    line_shift = np.where(in_possession, 8.0 * attack_sign, 0.0)   # push up on the ball
    targets[..., 0] += (0.45 * (rest_x + line_shift - targets[..., 0].mean(axis=-1))
                        )[..., None]

    # 3. width about the centre line, narrowed by directness
    lateral = targets[..., 1] - PITCH_W / 2.0
    targets[..., 1] = PITCH_W / 2.0 + lateral * width * (1.0 - 0.12 * directness)

    np.clip(targets[..., 0], 0.5, PITCH_L - 0.5, out=targets[..., 0])
    np.clip(targets[..., 1], 0.5, PITCH_W - 0.5, out=targets[..., 1])
    return targets


def press_targets(
    targets: np.ndarray,
    positions: np.ndarray,
    ball: np.ndarray,
    possessing_side: np.ndarray,
    press_idx: np.ndarray,
    collapse_near: float = 20.0,
    collapse_mid: float = 35.0,
) -> np.ndarray:
    """Override the shape target with "go and win the ball" for pressing players.

    The press trigger is a radius set by the press head. Only the nearest few
    defenders chase, so a gegenpress does not degenerate into all eleven players
    running at the ball — which would look absurd on the console and would break
    the calibration bands.

    Returns a modified copy of `targets`.
    """
    mods = modifier_tables()
    radius = mods["trigger_radius"][press_idx]                     # (B, 2)

    distance = np.linalg.norm(positions - ball[:, None, None, :], axis=-1)  # (B, 2, 11)

    # Only the defending side presses.
    defending = (possessing_side[:, None] != np.array([HOME, AWAY])[None, :])  # (B, 2)

    # Rank by proximity; the two closest are allowed to leave their slot, plus
    # a third under a gegenpress.
    order = np.argsort(distance, axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(11)[None, None, :].repeat(
        distance.shape[0], 0).repeat(2, 1), axis=-1)

    # A defence collapses as the ball nears its own goal: in the defensive third
    # essentially everyone gets behind the ball, and inside the box the whole
    # block converges. Without this the box stayed empty, attackers strolled to
    # within two metres, and the median shot was worth 0.30 xG.
    own_goal_x = np.array([0.0, PITCH_L], dtype=np.float32)[None, :]
    danger = np.abs(ball[:, None, 0] - own_goal_x)                 # (B, 2)
    extra = np.where(danger < collapse_near, 5, np.where(danger < collapse_mid, 3, 0))

    n_pressers = 2 + (press_idx == 2).astype(np.int64) + extra     # (B, 2)
    # The trigger radius must also widen, or the extra defenders are selected and
    # then immediately filtered out by the press-intensity radius.
    effective_radius = np.maximum(radius, np.where(danger < collapse_mid, 30.0, 0.0))

    chases = (
        (rank < n_pressers[..., None])
        & (distance < effective_radius[..., None])
        & defending[..., None]
    )
    chases[:, :, 0] = False                                        # never the keeper

    out = targets.copy()
    out[chases] = np.broadcast_to(ball[:, None, None, :], targets.shape)[chases]
    return out


def step_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    max_speed: np.ndarray,
    stamina: np.ndarray,
    dt: float,
    accel: float,
    speed_at_zero_stamina: float,
    closing_speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Steer every player toward their target under an acceleration limit.

    Speed scales with stamina, so a gegenpressing side that has emptied the tank
    by minute 70 physically cannot sustain the press — which is the mechanism
    that makes the fatigue term in the reward mean something.
    """
    delta = targets - positions
    distance = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / np.maximum(distance, 1e-6)

    effective = max_speed * (speed_at_zero_stamina
                             + (1.0 - speed_at_zero_stamina) * stamina)
    # `closing_speed` is per-player (B, 2, 11): it carries the press multiplier
    # and, for the ball carrier, the penalty for running with the ball at his
    # feet. Without that penalty a dribbler simply outruns the defence to goal.
    effective = effective * closing_speed

    # Ease off within a metre of the target so players settle instead of jittering.
    desired = direction * np.minimum(effective[..., None], distance / max(dt, 1e-6))

    dv = desired - velocities
    dv_norm = np.linalg.norm(dv, axis=-1, keepdims=True)
    dv = dv * np.minimum(1.0, accel * dt / np.maximum(dv_norm, 1e-6))

    new_velocities = velocities + dv
    new_positions = positions + new_velocities * dt

    np.clip(new_positions[..., 0], 0.0, PITCH_L, out=new_positions[..., 0])
    np.clip(new_positions[..., 1], 0.0, PITCH_W, out=new_positions[..., 1])
    return new_positions, new_velocities


def update_stamina(
    stamina: np.ndarray,
    velocities: np.ndarray,
    max_speed: np.ndarray,
    press_idx: np.ndarray,
    possessing_side: np.ndarray,
    dt: float,
    drain_base: float,
    recovery: float,
) -> np.ndarray:
    """Drain with effort, recover when the team has the ball.

    Convention: 1 = FRESH, 0 = EXHAUSTED — the inverse of GRF's `tired_factor`,
    whose sign the original draft inverted (design note 1, row 6).
    """
    mods = modifier_tables()
    drain_mult = mods["stamina_drain"][press_idx][..., None]        # (B, 2, 1)

    effort = np.linalg.norm(velocities, axis=-1) / np.maximum(max_speed, 1e-6)
    in_possession = (possessing_side[:, None] == np.array([HOME, AWAY])[None, :])

    drained = stamina - drain_base * drain_mult * (0.35 + effort**2) * dt
    recovered = drained + recovery * dt * in_possession[..., None] * (1.0 - effort)
    return np.clip(recovered, 0.02, 1.0)

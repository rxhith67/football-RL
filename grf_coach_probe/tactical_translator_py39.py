"""Three fixed high-level instructions translated into sparse GRF V2 actions.

The only interventions are movement and sprint for at most two outfield players.
Everything else remains on GRF's built-in AI. Selection is held for each
continuous opponent possession; it is never recalculated every frame.
"""

import itertools
import math

from gfootball.env import football_action_set as action_set


V2 = action_set.action_set_v2
ACTION = {str(a): i for i, a in enumerate(V2)}
AI = ACTION["builtin_ai"]
SPRINT = ACTION["sprint"]
RELEASE_DIRECTION = ACTION["release_direction"]
RELEASE_SPRINT = ACTION["release_sprint"]
DIRECTION_ACTIONS = (
    ("right", (1.0, 0.0)),
    # GRF's exposed observation y-axis is inverted relative to the native
    # controller's top/bottom labels (verified in live player trajectories).
    ("top_right", (1.0, -1.0)),
    ("top", (0.0, -1.0)),
    ("top_left", (-1.0, -1.0)),
    ("left", (-1.0, 0.0)),
    ("bottom_left", (-1.0, 1.0)),
    ("bottom", (0.0, 1.0)),
    ("bottom_right", (1.0, 1.0)),
)
assert len(V2) == 20 and AI == 19
assert all(name in ACTION for name, _ in DIRECTION_ACTIONS)


def norm(x, y):
    return math.hypot(x, y)


def distance(a, b):
    return norm(a[0] - b[0], a[1] - b[1])


def direction_action(position, target, tolerance=0.022):
    dx, dy = target[0] - position[0], target[1] - position[1]
    if norm(dx, dy) <= tolerance:
        return RELEASE_DIRECTION
    best_name = max(DIRECTION_ACTIONS,
                    key=lambda item: (dx * item[1][0] + dy * item[1][1]) /
                    norm(*item[1]))[0]
    return ACTION[best_name]


def cover_targets(ball):
    """Two distinct points between the carrier and the left goal."""
    bx, by = float(ball[0]), float(ball[1])
    goal = (-0.98, 0.0)
    near = (max(-0.92, goal[0] + 0.72 * (bx - goal[0])), 0.72 * by)
    deep = (max(-0.94, goal[0] + 0.43 * (bx - goal[0])), 0.43 * by)
    return near, deep


class TacticalTranslator:
    def __init__(self, policy):
        assert policy in ("normal", "press", "cover", "adaptive")
        self.policy = policy
        self.tick = 0
        self.active = {}  # action-slot -> target index
        self.release = {}  # action-slot -> 2, then 1
        self.cooldown = 0
        self.neutral_frames = 0
        self.last_effective_policy = "normal"
        self.engagement_start = -1000
        self.engagements = []

    @staticmethod
    def adaptive_choice(row):
        # Transparent, predeclared heuristic; no tuning against outcomes.
        bx, by = float(row["ball"][0]), float(row["ball"][1])
        return "cover" if bx < -0.40 or abs(by) >= 0.23 else "press"

    @staticmethod
    def _select_slots(obs, policy):
        row = obs[0]
        ball = row["ball"]
        carrier = int(row["ball_owned_player"])
        carrier_pos = row["right_team"][carrier]
        candidates = [slot for slot, item in enumerate(obs)
                      if 0 < int(item["active"]) < len(row["left_team"])
                      and not (int(row["ball_owned_team"]) == 0
                               and int(item["active"]) == int(row["ball_owned_player"]))]
        if policy == "press":
            chosen = sorted(candidates,
                            key=lambda slot: (distance(row["left_team"][int(obs[slot]["active"])], carrier_pos), slot))[:2]
            return {slot: 0 for slot in chosen}
        targets = cover_targets(ball)
        best = None
        for slots in itertools.permutations(candidates, 2):
            cost = sum(distance(row["left_team"][int(obs[slot]["active"])], target)
                       + (0.12 if row["left_team"][int(obs[slot]["active"])][0] > ball[0] else 0.0)
                       for slot, target in zip(slots, targets))
            candidate = (cost, slots)
            if best is None or candidate < best:
                best = candidate
        return {best[1][0]: 0, best[1][1]: 1} if best else {}

    def _targets(self, row, effective_policy):
        if effective_policy == "press":
            carrier = int(row["ball_owned_player"])
            target = (row["right_team"][carrier]
                      if int(row["ball_owned_team"]) == 1 and carrier >= 0
                      else row["ball"])
            return (target, target)
        return cover_targets(row["ball"])

    def _close_engagement(self, reason):
        if self.engagements and "end_tick" not in self.engagements[-1]:
            entry = self.engagements[-1]
            entry["end_tick"] = self.tick - 1
            entry["duration_steps"] = self.tick - entry["start_tick"]
            entry["end_reason"] = reason

    def finish(self):
        if self.active and self.engagements and "end_tick" not in self.engagements[-1]:
            entry = self.engagements[-1]
            entry["end_tick"] = self.tick
            entry["duration_steps"] = self.tick - entry["start_tick"] + 1
            entry["end_reason"] = "rollout_end"

    def actions(self, obs):
        self.tick += 1
        row = obs[0]
        actions = [AI] * len(obs)
        meta = {"tick": self.tick, "overridden": [], "targets": {},
                "commands": {}, "effective_policy": "normal"}
        ownership = int(row["ball_owned_team"])
        if ownership == 0:
            # Native builtin_ai resets the external keyboard (including
            # sprint/direction). Hand control back immediately: the selected
            # defender may now be our ball carrier.
            if self.active:
                self._close_engagement("left_possession")
                self.cooldown = 12
            self.active = {}
            self.release = {}
            self.neutral_frames = 0
            return actions, meta
        opponent_possesses = ownership == 1
        if opponent_possesses:
            self.neutral_frames = 0
        elif ownership == -1 and self.active:
            self.neutral_frames += 1
        else:
            self.neutral_frames = 0
        effective = (self.adaptive_choice(row) if self.policy == "adaptive"
                     else self.policy)
        if (not opponent_possesses
                and not (ownership == -1 and self.active
                         and self.neutral_frames <= 8)):
            effective = "normal"
        elif ownership == -1 and self.active:
            effective = self.last_effective_policy
        if self.policy == "normal":
            effective = "normal"
        if (self.active and ownership != 0
                and effective != self.last_effective_policy
                and self.tick - self.engagement_start < 30):
            effective = self.last_effective_policy

        # A tactic changes only at a new possession, and all previous external
        # inputs are explicitly released when that possession ends.
        if self.active and effective != self.last_effective_policy:
            self._close_engagement("possession_ended" if effective == "normal"
                                   else "instruction_changed")
            for slot in self.active:
                self.release[slot] = 2
            self.active = {}
            self.cooldown = 12

        for slot in list(self.release):
            phase = self.release[slot]
            actions[slot] = RELEASE_DIRECTION if phase == 2 else RELEASE_SPRINT
            meta["commands"][str(slot)] = ("release_direction" if phase == 2
                                           else "release_sprint")
            if phase == 1:
                del self.release[slot]
            else:
                self.release[slot] = 1

        if self.cooldown > 0:
            self.cooldown -= 1
        if (effective != "normal" and not self.active and not self.release
                and self.cooldown == 0):
            self.active = self._select_slots(obs, effective)
            self.engagement_start = self.tick
            self.engagements.append({"start_tick": self.tick,
                                     "policy": effective,
                                     "players": [int(obs[s]["active"])
                                                 for s in sorted(self.active)]})
            self.last_effective_policy = effective
            self._just_started = True
        else:
            self._just_started = False

        if self.active:
            targets = self._targets(row, self.last_effective_policy)
            meta["effective_policy"] = self.last_effective_policy
            for slot, target_index in self.active.items():
                player = int(obs[slot]["active"])
                assert player != 0  # Never override the goalkeeper.
                target = targets[target_index]
                action = (SPRINT if self._just_started else
                          direction_action(row["left_team"][player], target))
                actions[slot] = action
                meta["overridden"].append(player)
                meta["targets"][str(player)] = [float(target[0]), float(target[1])]
                meta["commands"][str(slot)] = str(V2[action])
        return actions, meta

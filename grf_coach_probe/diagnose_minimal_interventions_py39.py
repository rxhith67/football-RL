"""Paired, one-burst GRF defensive override diagnostic. No RL training.

The tested burst begins at the first observed opponent possession. It is never
repeated in a rollout. Duration values were fixed before outcome inspection.
"""

import argparse
import json
from pathlib import Path

import gfootball.env as football_env

from evaluate_tactics_py39 import evaluate
from tactical_translator_py39 import (
    AI, SPRINT, TacticalTranslator, V2, direction_action,
)


class OneBurstTranslator:
    def __init__(self, variant):
        tactic, count, limit = variant.split("_")
        assert tactic in ("press", "cover")
        assert count in ("1", "2")
        assert limit in ("12", "36", "cont")
        self.policy = tactic
        self.player_count = int(count)
        self.duration_limit = None if limit == "cont" else int(limit)
        self.tick = 0
        self.started = False
        self.active = {}
        self.neutral_frames = 0
        self.engagements = []

    def _close(self, reason):
        if self.active and self.engagements and "end_tick" not in self.engagements[-1]:
            entry = self.engagements[-1]
            entry["end_tick"] = self.tick - 1
            entry["duration_steps"] = self.tick - entry["start_tick"]
            entry["end_reason"] = reason
        self.active = {}

    def finish(self):
        if self.active and self.engagements and "end_tick" not in self.engagements[-1]:
            entry = self.engagements[-1]
            entry["end_tick"] = self.tick
            entry["duration_steps"] = self.tick - entry["start_tick"] + 1
            entry["end_reason"] = "rollout_end"

    def actions(self, obs):
        self.tick += 1
        row = obs[0]
        owner = int(row["ball_owned_team"])
        actions = [AI] * len(obs)
        meta = {"tick": self.tick, "overridden": [], "targets": {},
                "commands": {}, "effective_policy": "normal"}
        if self.active:
            if owner == -1:
                self.neutral_frames += 1
            else:
                self.neutral_frames = 0
            if owner == 0:
                self._close("left_possession")
            elif self.neutral_frames > 8:
                self._close("possession_ended")
            elif (self.duration_limit is not None
                  and self.tick - self.engagements[-1]["start_tick"]
                  >= self.duration_limit):
                self._close("duration_limit")

        just_started = False
        if not self.started and owner == 1:
            chosen = TacticalTranslator._select_slots(obs, self.policy)
            self.active = dict(sorted(chosen.items(), key=lambda item: item[1])
                               [:self.player_count])
            self.started = True
            just_started = True
            self.engagements.append({
                "start_tick": self.tick, "policy": self.policy,
                "players": [int(obs[slot]["active"]) for slot in self.active]})

        if self.active:
            targets = TacticalTranslator(self.policy)._targets(row, self.policy)
            meta["effective_policy"] = self.policy
            for slot, target_index in self.active.items():
                player = int(obs[slot]["active"])
                assert player != 0
                target = targets[target_index]
                action = (SPRINT if just_started else
                          direction_action(row["left_team"][player], target))
                actions[slot] = action
                meta["overridden"].append(player)
                meta["targets"][str(player)] = [float(target[0]), float(target[1])]
                meta["commands"][str(slot)] = str(V2[action])
        return actions, meta


def factory(variant):
    if variant in ("normal", "press", "cover"):
        return TacticalTranslator(variant)
    return OneBurstTranslator(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--variants", default=(
        "normal,press,cover,press_1_cont,cover_1_cont,"
        "press_1_12,press_1_36,cover_1_12,cover_1_36,"
        "press_2_12,cover_2_12"))
    parser.add_argument("--horizon", type=int, default=200)
    args = parser.parse_args()
    states_dir = Path(args.states)
    manifest = json.loads((states_dir / "manifest.json").read_text())
    ids = set(args.ids.split(",")) if args.ids else None
    metas = [m for m in manifest["states"] if ids is None or m["id"] in ids]
    variants = args.variants.split(",")
    env = football_env.create_environment(
        env_name=manifest["scenario"], representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=11,
        other_config_options={"action_set": "v2"})
    results = []
    try:
        env.reset()
        for meta in metas:
            snapshot = (states_dir / (meta["id"] + ".state")).read_bytes()
            for variant in variants:
                row = evaluate(env, snapshot, meta, variant, args.horizon,
                               False, factory)
                results.append(row)
                print(json.dumps({"state": meta["id"], "variant": variant,
                                  "recovery": row["time_to_recovery"],
                                  "progress": row["opponent_progress_toward_goal"],
                                  "goals_conceded": row["goals_conceded"],
                                  "override_frames": sum(
                                      row["override_frames_by_player"].values())}),
                      flush=True)
    finally:
        env.close()
    output = Path(args.out)
    output.write_text(json.dumps({
        "scenario": manifest["scenario"],
        "state_ids": [m["id"] for m in metas],
        "variants": variants, "horizon": args.horizon,
        "results": results}, indent=2) + "\n")
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()

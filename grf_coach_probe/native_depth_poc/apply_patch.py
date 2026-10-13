"""Apply the one-parameter coach-depth patch to an ISOLATED copy of the GRF engine.

usage: python3 apply_patch.py <path-to-copy>/third_party/gfootball_engine
Never run this against the original build directory.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert "grf_native_poc" in str(root), "refusing to patch anything but the isolated copy"


def edit(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    assert s.count(old) == count, (rel, old, s.count(old))
    p.write_text(s.replace(old, new))


# 1. TeamAIController: one coach override of position_defense_depth_factor
edit("src/onthepitch/teamAIcontroller.hpp",
     "    void UpdateTactics();\n",
     "    void UpdateTactics();\n"
     "    // PoC: coach override of position_defense_depth_factor (0..1). value < 0 clears it.\n"
     "    // Not part of ProcessState on purpose, so existing snapshots stay loadable.\n"
     "    void SetCoachDefenseDepth(float value);\n"
     "    float GetLiveDefenseDepth() const { return liveTeamTactics.GetReal(\"position_defense_depth_factor\"); }\n")
edit("src/onthepitch/teamAIcontroller.hpp",
     "    float offensivenessBias = 0.0f;\n",
     "    float offensivenessBias = 0.0f;\n"
     "    float coachDefenseDepth = -1.0f;  // PoC override, < 0 == none\n")

edit("src/onthepitch/teamAIcontroller.cpp",
     "void TeamAIController::Reset() {\n",
     "void TeamAIController::SetCoachDefenseDepth(float value) {\n"
     "  coachDefenseDepth = value < 0.0f ? -1.0f : clamp(value, 0.0f, 1.0f);\n"
     "  // apply immediately; UpdateTactics() re-applies it every 1000 ms of match time\n"
     "  if (coachDefenseDepth >= 0.0f) liveTeamTactics.Set(\"position_defense_depth_factor\", coachDefenseDepth);\n"
     "  // clearing deliberately does NOT call UpdateTactics(): live values re-sync on the engine own 1000 ms schedule\n"
     "}\n\n"
     "void TeamAIController::Reset() {\n")
# re-apply override at the end of UpdateTactics (it rebuilds liveTeamTactics from base every second)
edit("src/onthepitch/teamAIcontroller.cpp",
     "    iter++;\n  }\n}\n\nvoid TeamAIController::SetCoachDefenseDepth",
     "    iter++;\n  }\n"
     "  if (coachDefenseDepth >= 0.0f) liveTeamTactics.Set(\"position_defense_depth_factor\", coachDefenseDepth);\n"
     "}\n\nvoid TeamAIController::SetCoachDefenseDepth")

# 2. GameEnv: two thin methods (set + read-back of the live value)
edit("src/game_env.hpp",
     "  void tracker_setup(long start, long end) { GetTracker()->setup(start, end); }\n",
     "  void tracker_setup(long start, long end) { GetTracker()->setup(start, end); }\n"
     "  // PoC: team_id 0 = left, 1 = right. value in [0,1]; value < 0 clears the override.\n"
     "  void set_coach_defense_depth(int team_id, float value);\n"
     "  float get_live_defense_depth(int team_id);\n")
edit("src/game_env.cpp",
     "bool GameEnv::sticky_action_state(int action, bool left_team, int player) {\n",
     "void GameEnv::set_coach_defense_depth(int team_id, float value) {\n"
     "  SetGame(this);\n"
     "  GetTracker()->setDisabled(true);\n"
     "  if (team_id == 0 || team_id == 1) {\n"
     "    GetGameTask()->GetMatch()->GetTeam(team_id)->GetController()->SetCoachDefenseDepth(value);\n"
     "  }\n"
     "  GetTracker()->setDisabled(false);\n"
     "}\n\n"
     "float GameEnv::get_live_defense_depth(int team_id) {\n"
     "  SetGame(this);\n"
     "  GetTracker()->setDisabled(true);\n"
     "  float v = GetGameTask()->GetMatch()->GetTeam(team_id)->GetController()->GetLiveDefenseDepth();\n"
     "  GetTracker()->setDisabled(false);\n"
     "  return v;\n"
     "}\n\n"
     "bool GameEnv::sticky_action_state(int action, bool left_team, int player) {\n")

# 3. Python binding
edit("ai.cpp",
     "      .def(\"tracker_setup\", &GameEnv::tracker_setup);",
     "      .def(\"tracker_setup\", &GameEnv::tracker_setup)\n"
     "      .def(\"set_coach_defense_depth\", &GameEnv::set_coach_defense_depth)\n"
     "      .def(\"get_live_defense_depth\", &GameEnv::get_live_defense_depth);")
print("patched", root)

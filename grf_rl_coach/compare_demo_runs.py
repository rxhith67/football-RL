"""Check that two demo runs (same checkpoint, same seed) are identical.  usage: bash run.sh compare_demo_runs.py A B"""
import json
import sys

a, b = (json.load(open(f"{d}/timeline.json")) for d in sys.argv[1:3])
key = lambda d: [(x["action"], x["score_end"], x["r_goal"], x["r_shape"], x["team_length"], x["back_line_x"]) for x in d["decisions"]]
print("identical decisions, scores, rewards and team shape:", key(a) == key(b))
print("final score", a["final_score"], b["final_score"], "| events", a["events"])

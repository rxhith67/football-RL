"""Run ONE full GRF match controlled by a saved coach, record a GRF video, and write a
decision timeline (JSON + CSV + figure) plus an annotated video showing the coach's tactic.
usage: bash run.sh demo.py --policy q --ckpt checkpoints/q_main.npz --seed 900001 --out demo/q_seed900001

Add --native-render to record GRF's graphical football view. This is much slower
than headless recording and skips the extra text-overlay video.
The built-in football AI controls all 11 players; the coach only sets team-level defensive depth."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from coach_env import GRFDepthCoachEnv, ACTION_NAMES, FEATURE_NAMES, ACTION_DEPTH, AI, NATIVE_DEFAULT_DEPTH
import policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["q", "dqn", "hand", "native"], default="q")
    ap.add_argument("--ckpt", default="checkpoints/q_main.npz")
    ap.add_argument("--seed", type=int, default=900001)
    ap.add_argument("--out", default="demo/run")
    ap.add_argument("--native-render", action="store_true", help="Record actual GRF rendered frames")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    e = GRFDepthCoachEnv(write_video=True, logdir=out, render_video=args.native_render)
    e.noop_calls = args.policy == "native"
    if args.policy == "q":
        q = np.load(args.ckpt)["q"]
    elif args.policy == "dqn":
        import torch
        from train_dqn import make_net, np_weights
        net = make_net(); net.load_state_dict(torch.load(args.ckpt)["net"]); w = np_weights(net)
    sent = {"calls": 0, "non_builtin_ai": 0}
    real_step = e.env.step
    def spy(actions):                              # every action list sent to GRF is inspected
        sent["calls"] += 1
        sent["non_builtin_ai"] += sum(1 for a in actions if a != AI)
        return real_step(actions)
    e.env.step = spy
    f = e.reset(seed=args.seed)
    done, rows, step0 = False, [], 0
    while not done:
        s = e.discretize(f)
        a = (policies.q_action(q, s) if args.policy == "q" else policies.dqn_action(w, f) if args.policy == "dqn"
             else policies.hand_coach(f) if args.policy == "hand" else 1)
        feats = dict(zip(FEATURE_NAMES, f.round(3).tolist()))
        f2, r, done, info = e.step(a)
        rows.append({"decision": len(rows) + 1, "engine_step_start": step0, "engine_steps": info["engine_steps"],
                     "action": ACTION_NAMES[a], "applied_depth": info["applied_depth"], "live_depth_end": info["live_depth"],
                     "ball_x_start": feats["ball_x"], "owner_start": ("ours" if feats["own_poss"] else "opp" if feats["opp_poss"] else "loose"),
                     "score_end": info["score"], "r_goal": info["r_goal"], "r_shape": round(info["r_shape"], 4),
                     "team_length": round(info["team_length"], 3), "back_line_x": round(info["back_line_x"], 3),
                     "possession_gains": info["possession_gains"], "goals_for": info["goals_for"], "goals_against": info["goals_against"],
                     "ball_x_end": round(float(e.prev["ball"][0]), 3)})
        step0 += info["engine_steps"]; f = f2
    final = rows[-1]["score_end"]
    e.close()
    expected = {"compact": 0.0, "stretched": 1.0, "native_default": NATIVE_DEFAULT_DEPTH}
    reached = [abs((r["live_depth_end"] if r["live_depth_end"] is not None else NATIVE_DEFAULT_DEPTH) - expected[r["action"]]) < 1e-4 for r in rows]
    events = []
    for r in rows:
        if r["goals_for"]: events.append({"decision": r["decision"], "engine_step_end": r["engine_step_start"] + r["engine_steps"], "event": f"GOAL FOR us x{r['goals_for']}"})
        if r["goals_against"]: events.append({"decision": r["decision"], "engine_step_end": r["engine_step_start"] + r["engine_steps"], "event": f"GOAL AGAINST x{r['goals_against']}"})
    verification = {"decisions": len(rows), "engine_steps_total": step0,
                    "live_native_depth_matches_coach_action_at_end_of_every_decision": bool(all(reached)),
                    "decisions_where_live_depth_matched": int(sum(reached)),
                    "engine_step_calls": sent["calls"], "non_builtin_ai_actions_sent_to_players": sent["non_builtin_ai"],
                    "action_counts": {n: sum(1 for r in rows if r["action"] == n) for n in ACTION_NAMES},
                    "total_reward_raw_goals": sum(r["r_goal"] for r in rows), "total_reward_shaping": round(sum(r["r_shape"] for r in rows), 4),
                    "total_reward": round(sum(r["r_goal"] + r["r_shape"] for r in rows), 4)}
    json.dump({"policy": args.policy, "checkpoint": args.ckpt if args.policy in ("q", "dqn") else None, "seed": args.seed,
               "final_score": final, "verification": verification, "events": events, "decisions": rows}, open(out / "timeline.json", "w"), indent=1)
    with open(out / "timeline.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    # --- figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"compact": "#3b82f6", "native_default": "#cbd5e1", "stretched": "#f97316"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for r in rows:
        ax[0].axvspan(r["engine_step_start"], r["engine_step_start"] + r["engine_steps"], color=col[r["action"]], alpha=.5, lw=0)
    ax[0].plot([r["engine_step_start"] for r in rows], [r["team_length"] for r in rows], "k.-", lw=1)
    ax[0].set_ylabel("team length (shape)")
    prev = [0, 0]
    for r in rows:
        if r["score_end"] != prev:
            ax[0].axvline(r["engine_step_start"] + r["engine_steps"], color="green" if r["score_end"][0] > prev[0] else "red", ls="--")
            ax[0].text(r["engine_step_start"] + r["engine_steps"], ax[0].get_ylim()[1] * .95, f"{r['score_end'][0]}-{r['score_end'][1]}", fontsize=8)
            prev = r["score_end"]
    ax[1].step([r["engine_step_start"] for r in rows], [r["r_goal"] + r["r_shape"] for r in rows], where="post", color="#555")
    ax[1].set_ylabel("reward"); ax[1].set_xlabel("engine step")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=c, label=k, alpha=.5) for k, c in col.items()], loc="upper right", fontsize=7, ncol=3)
    ax[0].set_title(f"Coach timeline ({args.policy}, seed {args.seed}, final score {final[0]}-{final[1]})")
    plt.tight_layout(); plt.savefig(out / "timeline.png", dpi=150)
    # --- annotated video (overlay coach tactic)
    avis = sorted(out.glob("episode_done_*.avi")) or sorted(out.glob("*.avi"))
    if avis and not args.native_render:
        import cv2
        cap = cv2.VideoCapture(str(avis[0])); fps = cap.get(cv2.CAP_PROP_FPS) or 10
        W, H = int(cap.get(3)), int(cap.get(4)); n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vw = cv2.VideoWriter(str(out / "annotated.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))
        starts = [r["engine_step_start"] for r in rows]
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            k = max(0, min(len(rows) - 1, int(np.searchsorted(starts, idx, side="right") - 1)))
            r = rows[k]
            cv2.rectangle(fr, (0, 0), (W, 34), (20, 20, 20), -1)
            cv2.putText(fr, f"COACH: {r['action'].upper()}  decision {r['decision']}/{len(rows)}  step {idx}  score {r['score_end'][0]}-{r['score_end'][1]}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80) if r["action"] == "native_default" else (60, 160, 255), 2)
            vw.write(fr); idx += 1
        vw.release(); cap.release()
        print("video frames", n_frames, "annotated", idx)
    print("final score", final, "decisions", len(rows), "files", sorted(p.name for p in out.iterdir()))
    print(json.dumps(verification, indent=1))


if __name__ == "__main__":
    main()

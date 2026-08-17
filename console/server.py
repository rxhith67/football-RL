"""GAFFER Console backend — FastAPI over the replay format.

Two-process design (design note 4): the browser animates a *pre-computed*
replay, so real-time rendering never fights a server round-trip. Scrubbing a
recorded match is also a better demo than live animation — on stage you can jump
straight to the interesting moment.

Endpoints
---------
    GET  /                  the console page
    POST /simulate          run a match, return the full replay JSON
    POST /step              duel mode: human picks the opposing tactic for one block
    POST /whatif            rewind to minute X, force a tactic, re-simulate N times

The server consumes only `pitchsim.replay`'s versioned schema, never engine
internals, which is what lets the whole demo run offline from cached replays
(Gate 4).

    uvicorn console.server:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.archetypes import archetype_table  # noqa: E402
from pitchsim.engine import PitchSimEngine  # noqa: E402
from pitchsim.types import (  # noqa: E402
    BLOCK_HEIGHTS,
    FORMATIONS,
    PRESS_LEVELS,
    TEMPOS,
    Side,
    Tactic,
)

app = FastAPI(title="GAFFER Console")
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class SimulateRequest(BaseModel):
    seed: int = 0
    home: dict[str, str] = {}
    away: dict[str, str] = {}
    minutes: float = 90.0


class WhatIfRequest(BaseModel):
    seed: int = 0
    minute: float = 60.0
    forced: dict[str, str] = {}
    rollouts: int = 100


def _tactic(spec: dict[str, str]) -> Tactic:
    return Tactic.from_names(
        formation=spec.get("formation", "4-4-2"),
        block_height=spec.get("block_height", "mid"),
        press=spec.get("press", "balanced"),
        tempo=spec.get("tempo", "balanced"),
    )


@app.get("/")
def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"error": "console/static/index.html not found"}


@app.get("/options")
def options():
    """Everything the UI needs to build its dropdowns."""
    _, labels, _ = archetype_table()
    return {
        "formation": list(FORMATIONS),
        "block_height": list(BLOCK_HEIGHTS),
        "press": list(PRESS_LEVELS),
        "tempo": list(TEMPOS),
        "archetypes": labels,
    }


@app.post("/simulate")
def simulate(req: SimulateRequest):
    sim = PitchSimEngine()
    sim.reset(seed=req.seed, home_tactic=_tactic(req.home),
              away_tactic=_tactic(req.away), record=True)
    while not sim.done and sim.state.minute < req.minutes:
        segment = sim.advance(5.0)
        for side in (Side.HOME, Side.AWAY):
            sim._builder.add_epoch(
                minute=sim.state.minute,
                side=int(side),
                tactic=sim.state.tactics[side].as_names(),
                trigger=segment.trigger,
            )
    replay = sim.replay()
    # Say plainly who is who and who (if anyone) is actually choosing. In this
    # endpoint the tactics are FIXED from the request — no trained policy is
    # driving them yet — and the console must not imply otherwise.
    replay["meta"]["controller"] = "fixed tactics from the request (no policy)"
    replay["meta"]["home_label"] = "GAFFER (you) — red, attacking left to right"
    replay["meta"]["away_label"] = "Opponent — blue, attacking right to left"
    return replay


@app.post("/whatif")
def whatif(req: WhatIfRequest):
    """Rewind to `minute`, force a tactic, re-simulate, return the win-prob shift.

    The counterfactual panel: the same seed replays identically up to the rewind
    point, so any difference afterwards is attributable to the forced decision.
    """
    def run(force: bool) -> tuple[int, int, float]:
        sim = PitchSimEngine()
        sim.reset(seed=req.seed, record=False)
        while not sim.done:
            if force and sim.state.minute >= req.minute:
                sim.set_tactic(Side.HOME, _tactic(req.forced))
                force = False
            sim.advance(5.0)
        result = sim.result()
        return result.score[0], result.score[1], result.xg[0] - result.xg[1]

    baseline = [run(False) for _ in range(req.rollouts)]
    forced = [run(True) for _ in range(req.rollouts)]

    def win_rate(rows):
        return float(np.mean([1.0 if h > a else (0.5 if h == a else 0.0)
                              for h, a, _ in rows]))

    base_wp, forced_wp = win_rate(baseline), win_rate(forced)
    return {
        "minute": req.minute,
        "rollouts": req.rollouts,
        "baseline_win_prob": base_wp,
        "forced_win_prob": forced_wp,
        "delta": forced_wp - base_wp,
        "baseline_xg_diff": float(np.mean([x for _, _, x in baseline])),
        "forced_xg_diff": float(np.mean([x for _, _, x in forced])),
    }


class PolicyRequest(BaseModel):
    seed: int = 0
    checkpoint: str = "stage0_bayesian_smdp_seed0"


@app.post("/simulate_policy")
def simulate_policy(req: PolicyRequest):
    """Run a match with the TRAINED COACH choosing tactics every five minutes.

    This is the endpoint that makes the console show what the project is actually
    about. `/simulate` holds one tactic for ninety minutes; here the policy sees
    the observation at each decision epoch — including its belief over the
    opponent's archetype — and picks a fresh tactic, so the bench panel and the
    belief bars finally have something real to display.

    Falls back to a legal-random coach when no checkpoint exists, and says so in
    `meta.controller` rather than pretending a policy was involved.
    """
    from agents.baselines import RandomCoach
    from agents.smdp_ppo import SMDPMaskablePPO
    from envs.coach_env import VecCoachEnv
    from pitchsim.replay import ReplayBuilder

    env = VecCoachEnv(n_envs=1, belief_mode="bayesian", seed=req.seed)
    obs = env.reset()

    # Attach AFTER reset: the engine clears its recorders on reset.
    builder = ReplayBuilder(seed=req.seed, dt=env.engine.dt)
    env.engine.attach_recorder(0, builder)

    path = ROOT / "checkpoints" / f"{req.checkpoint}.zip"
    model, controller = None, ""
    if path.exists():
        model = SMDPMaskablePPO.load(str(path.with_suffix("")), device="cpu")
        controller = f"trained coach ({req.checkpoint})"
    else:
        fallback = RandomCoach(seed=req.seed)
        controller = "legal-random coach — no checkpoint found, train one first"

    _, labels, _ = archetype_table()
    episode, guard = None, 0
    last = (0.0, 0.0, 0, 0)      # xg_home, xg_away, shots_home, shots_away

    while guard < 60:
        guard += 1
        if model is not None:
            actions, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True)
        else:
            actions = fallback.act(obs, env)

        minute = float(env.engine.minutes()[0])
        builder.add_epoch(
            minute=minute,
            side=int(Side.HOME),
            tactic=Tactic.from_array(np.asarray(actions).reshape(-1, 5)[0]).as_names(),
            belief=[float(b) for b in env.belief[0]],
            trigger="scheduled",
        )
        # Snapshot the counters BEFORE stepping past full time. `env.step` resets
        # the episode internally once the match ends, which zeroes the engine's
        # counters — reading them afterwards reported 0 shots and 0.0 xG for a
        # match that plainly had goals in it.
        obs, _, dones, infos = env.step(actions)
        if not dones[0]:
            last = (float(env.engine.counters.xg[0, 0]),
                    float(env.engine.counters.xg[0, 1]),
                    int(env.engine.counters.shots[0, 0]),
                    int(env.engine.counters.shots[0, 1]))
            builder.add_xg(float(env.engine.minutes()[0]), last[0], last[1])
        else:
            episode = infos[0].get("episode")
            break

    replay = builder.as_dict()
    replay["meta"]["controller"] = controller
    replay["meta"]["home_label"] = "GAFFER (trained coach) — red, attacking left to right"
    replay["meta"]["away_label"] = "Opponent (scripted archetype) — blue"
    replay["meta"]["archetypes"] = labels
    if episode is not None:
        replay["result"] = {
            "score": list(episode["score"]),
            "xg": [last[0], last[1]],
            "shots": [last[2], last[3]],
            "possession_home": 0.5,
            "n_decisions": episode["l"],
            "seed": req.seed,
            "belief_correct": episode.get("belief_correct"),
        }
    return replay


@app.get("/health")
def health():
    return {"status": "ok", "engine": "PitchSimEngine", "is_stub": PitchSimEngine.IS_STUB}

"""Ablation sweep (Gate 3): belief modes and the SMDP discount.

Covers what the codebase actually implements today:
    * belief: blind vs bayesian vs oracle (envs/coach_env.py BELIEF_MODES)
    * gamma: gamma^tau (SMDP) vs flat gamma (train.py --flat-gamma)

Two sub-ablations from the Gate 3 checklist have NO implementation to run yet
and are declared, not silently skipped:
    * "recurrent" belief mode — would need an RNN belief filter; envs/belief.py
      only has the Bayesian filter.
    * "masking" — would need an unmasked-action policy variant; nothing in
      agents/smdp_ppo.py currently disables the action mask.
    * "shaping w sweep" — config/train.yaml declares shaping_w_sweep but no
      training entry point reads it; run() below does not touch reward shaping.
`ablations_report` reports these gaps explicitly rather than reporting a false
"all ablations run".
"""

from __future__ import annotations

import json
from pathlib import Path

from pitchsim.controller import config_fingerprint
from train import evaluate_agent, train

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "ablations.json"
STAGE0_PARENT = "stage0_bayesian_smdp_seed0"

IMPLEMENTED = ("belief", "gamma")
NOT_IMPLEMENTED = ("recurrent belief mode", "masking", "shaping-w sweep")


def run(steps: int = 15_000, n_envs: int = 16, n_matches: int = 30,
       seed: int = 0, parent: str = STAGE0_PARENT) -> dict:
    results = {"belief": {}, "gamma": {}}

    for mode in ("blind", "bayesian", "oracle"):
        model, env, _ = train(stage=1, steps=steps, n_envs=n_envs, belief_mode=mode,
                              seed=seed, smdp=True, verbose=0, init_checkpoint=parent,
                              tag_suffix="_abl")
        results["belief"][mode] = evaluate_agent(model, env, n_matches, seed)

    # gamma^tau at belief_mode=bayesian is the same run already done above —
    # only the flat-gamma ablation needs a fresh training run.
    results["gamma"]["gamma_tau"] = results["belief"]["bayesian"]
    model, env, _ = train(stage=1, steps=steps, n_envs=n_envs, belief_mode="bayesian",
                          seed=seed, smdp=False, verbose=0, init_checkpoint=parent,
                          tag_suffix="_abl")
    results["gamma"]["flat_gamma"] = evaluate_agent(model, env, n_matches, seed)

    blind, bayes, oracle = (results["belief"][m]["points_per_match"]
                            for m in ("blind", "bayesian", "oracle"))
    gap = oracle - blind
    results["oracle_gap"] = gap
    # "Belief recovers X% of the oracle gap" is only meaningful when the oracle
    # is actually ahead of blind. With gap < 0 the ratio divides one negative by
    # another and prints a healthy-looking percentage while the ordering is
    # backwards — exactly the number that must NOT reach a write-up unnoticed.
    results["belief_recovers"] = ((bayes - blind) / gap) if gap > 1e-9 else None
    results["ordering_as_expected"] = bool(oracle > blind)

    payload = {
        "config_fingerprint": config_fingerprint(),
        "steps": steps, "n_matches": n_matches, "seed": seed, "parent": parent,
        "implemented": list(IMPLEMENTED),
        "not_implemented": list(NOT_IMPLEMENTED),
        "results": results,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def ablations_report() -> tuple[bool, str]:
    """Gate 3 hook. True only means the IMPLEMENTED ablations ran — see detail."""
    if not RESULTS.exists():
        return False, "not yet run: python -m analysis.ablations"
    blob = json.loads(RESULTS.read_text(encoding="utf-8"))
    if blob.get("config_fingerprint") != config_fingerprint():
        return False, "stale evidence: rerun python -m analysis.ablations"
    r = blob["results"]
    belief = ", ".join(f"{m}={r['belief'][m]['points_per_match']:.2f}"
                       for m in ("blind", "bayesian", "oracle"))
    gamma = ", ".join(f"{k}={v['points_per_match']:.2f}" for k, v in r["gamma"].items())
    missing = ", ".join(blob["not_implemented"])

    # Overlapping CIs mean the belief ordering is not resolved at this sample
    # size, whichever way the point estimates happen to fall.
    lo_o, hi_o = r["belief"]["oracle"]["ppm_ci"]
    lo_b, hi_b = r["belief"]["blind"]["ppm_ci"]
    separated = lo_o > hi_b or lo_b > hi_o

    if not r.get("ordering_as_expected", True):
        return False, (f"belief ppm [{belief}] — ORDERING BACKWARDS: oracle is not ahead of "
                       f"blind (gap {r['oracle_gap']:+.2f}), so Pillar 1's claim is "
                       f"unsupported by this run; gamma ppm [{gamma}]")
    recovers = r.get("belief_recovers")
    detail = f"belief ppm [{belief}]"
    detail += f" recovers {100*recovers:.0f}%" if recovers is not None else " (oracle gap ~0)"
    if not separated:
        detail += " [CIs OVERLAP — ordering not resolved at this n]"
    return True, f"{detail}; gamma ppm [{gamma}] — NOT RUN (unimplemented): {missing}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ablation sweep (Gate 3)")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--matches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parent", default=STAGE0_PARENT)
    args = parser.parse_args()
    result = run(args.steps, args.n_envs, args.matches, args.seed, args.parent)
    print(json.dumps(result["results"], indent=2, default=float))


if __name__ == "__main__":
    main()

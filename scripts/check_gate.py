"""Phase gate checker.

The plan's phases advance on *evidence*, not on dates (plan section 6.2). This
script is what makes that real: each gate is a set of mechanical checks that
either pass or fail, so "are we ready for Phase 2?" has an answer nobody has to
argue about in a group meeting.

Gates for phases not yet reached report as PENDING with their criteria listed, so
the whole team can see what they are building toward.

Usage::

    python scripts/check_gate.py 0
    python scripts/check_gate.py 1
    python scripts/check_gate.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN, RED, YELLOW, GREY, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"


class Check:
    def __init__(self, name: str, passed: bool | None, detail: str = "") -> None:
        self.name, self.passed, self.detail = name, passed, detail

    def render(self) -> str:
        if self.passed is None:
            mark, colour = "PEND", YELLOW
        elif self.passed:
            mark, colour = "PASS", GREEN
        else:
            mark, colour = "FAIL", RED
        line = f"  {colour}[{mark}]{RESET} {self.name}"
        if self.detail:
            line += f"\n         {GREY}{self.detail}{RESET}"
        return line


# --------------------------------------------------------------------------
# Gate 0 — Foundations
# --------------------------------------------------------------------------


def _check_configs() -> Check:
    import yaml

    missing, loaded = [], {}
    for name in ("pitchsim.yaml", "tactics.yaml", "train.yaml"):
        path = ROOT / "config" / name
        if not path.exists():
            missing.append(name)
        else:
            loaded[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if missing:
        return Check("Configs frozen", False, f"missing: {', '.join(missing)}")

    heads = loaded["tactics.yaml"]["action_space"]["heads"]
    sizes = tuple(len(v) for v in heads.values())
    if sizes != (5, 3, 3, 3, 6):
        return Check("Configs frozen", False, f"head sizes {sizes} != (5, 3, 3, 3, 6)")
    return Check("Configs frozen", True, "pitchsim / tactics / train load; heads = (5,3,3,3,6)")


def _check_contract() -> Check:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else "no output"
    return Check("Contract tests pass", proc.returncode == 0, summary)


def _check_stub() -> Check:
    try:
        from pitchsim import StubPitchSim, Tactic

        sim = StubPitchSim()
        sim.reset(seed=0, record=True)
        while not sim.done:
            sim.advance(5.0)
        result = sim.result()
        from pitchsim import validate_replay

        problems = validate_replay(sim.replay())
        if problems:
            return Check("Stub PITCHSIM live", False, "; ".join(problems))
        return Check(
            "Stub PITCHSIM live",
            True,
            f"90' match ran, score {result.score[0]}-{result.score[1]}, "
            f"replay schema valid — owners B and C are unblocked",
        )
    except Exception as exc:
        return Check("Stub PITCHSIM live", False, f"{type(exc).__name__}: {exc}")


def _check_statsbomb() -> Check:
    from pitchsim.statsbomb import SHOTS_PARQUET

    if not SHOTS_PARQUET.exists():
        return Check("StatsBomb shots cached", False,
                     "run: python -m pitchsim.statsbomb --fetch")
    try:
        import pandas as pd

        df = pd.read_parquet(SHOTS_PARQUET)
        enough = len(df) >= 2000
        return Check(
            "StatsBomb shots cached", enough,
            f"{len(df)} shots from {df.match_id.nunique()} matches"
            + ("" if enough else " — want >= 2000 for a stable fit"),
        )
    except Exception as exc:
        return Check("StatsBomb shots cached", False, str(exc))


def _check_xg() -> Check:
    from pitchsim.xg import MIN_AUC, MODEL_PATH

    if not MODEL_PATH.exists():
        return Check(f"xG model AUC >= {MIN_AUC}", False, "run: python -m pitchsim.xg --fit")
    metrics = json.loads(MODEL_PATH.read_text(encoding="utf-8")).get("metrics") or {}
    auc = metrics.get("auc")
    if auc is None:
        return Check(f"xG model AUC >= {MIN_AUC}", False, "model has no recorded metrics")
    detail = f"held-out AUC = {auc:.4f} on {metrics.get('n_shots', '?')} shots"
    if "statsbomb_auc" in metrics:
        detail += f" (StatsBomb's own xG: {metrics['statsbomb_auc']:.4f})"
    return Check(f"xG model AUC >= {MIN_AUC}", auc >= MIN_AUC, detail)


def gate_0() -> list[Check]:
    return [_check_configs(), _check_stub(), _check_contract(),
            _check_statsbomb(), _check_xg()]


# --------------------------------------------------------------------------
# Gates 1-4 — declared now so the team can see what they are building toward
# --------------------------------------------------------------------------


def _pending(name: str, detail: str) -> Check:
    return Check(name, None, detail)


def gate_1() -> list[Check]:
    checks = [
        _pending("A match that looks like football",
                 "HUMAN JUDGEMENT: a football-literate person watching the animation "
                 "says yes, unprompted. If this fails, switch to Option B without debate."),
    ]
    try:
        from pitchsim.calibrate import calibration_report, performance_report  # type: ignore

        checks.append(Check("Calibration bands met", *calibration_report()))
        checks.append(Check("Throughput >= 300 matches/min", *performance_report()))
    except ImportError:
        checks += [
            _pending("Calibration bands met",
                     "goals 2.3-3.1 /match · shots 9-15 /team · xG/shot 0.08-0.13 · "
                     "pass completion 72-88% (pitchsim/calibrate.py, Phase 1)"),
            _pending("Throughput >= 300 matches/min", "32 vectorised envs on Colab CPU"),
        ]
    try:
        from analysis.separability import separability_report  # type: ignore

        checks.append(Check("Archetype separability", *separability_report()))
    except ImportError:
        checks.append(_pending(
            "Archetype separability >= 70%",
            "K-way confusion matrix on 15-min windows, no archetype below 50% recall. "
            "Voids Pillar 1 if it fails — verify BEFORE wiring belief to RL."))
    try:
        from analysis.egta import scripted_cycle_report  # type: ignore

        checks.append(Check("Non-transitive cycle exists", *scripted_cycle_report()))
    except ImportError:
        checks.append(_pending(
            "Non-transitive cycle exists",
            "K x K scripted round-robin, 200 matches/cell. Voids Pillar 2 if absent — "
            "find out now, not in Phase 3."))
    return checks


def gate_2() -> list[Check]:
    return [
        _pending("PPO beats random-tactic, non-overlapping 95% CIs", "Stage 0-1 curriculum"),
        _pending("Belief re-identifies a switch within 3 epochs",
                 "measured from the TRUE switch epoch, not from kickoff"),
        _pending("Elo rises across self-play generations", "Stage 2 PFSP league"),
        _pending("Console renders a real replay end to end", "broadcast pitch + xG race"),
    ]


def gate_3() -> list[Check]:
    return [
        _pending("EGTA payoff matrix + Nash/alpha-Rank complete", "analysis/egta.py"),
        _pending("Exploitability probe at declared budget",
                 "200k macro-steps, 3 seeds, 500 eval matches — budget goes in the caption"),
        _pending("All ablations run",
                 "belief/blind/oracle/recurrent · shaping w sweep · masking · gamma^tau"),
        _pending("FEATURE FREEZE 15 Oct", "every number in the report exists on disk"),
    ]


def gate_4() -> list[Check]:
    return [
        _pending("Demo runs offline on the presenting laptop", "network cable pulled"),
        _pending("Rehearsed 3x on the actual hardware", ""),
        _pending("Viva Q&A bank complete",
                 "SMDP discounting · potential-based shaping · non-transitivity · POMDP belief"),
    ]


GATES = {
    0: ("Foundations & de-risking", gate_0),
    1: ("Environment — THE go/no-go", gate_1),
    2: ("Learning", gate_2),
    3: ("Science", gate_3),
    4: ("Consolidation", gate_4),
}


def run_gate(number: int) -> bool:
    title, fn = GATES[number]
    print(f"\n{'=' * 68}\n  GATE {number} — {title}\n{'=' * 68}")
    checks = fn()
    for check in checks:
        print(check.render())

    failed = [c for c in checks if c.passed is False]
    pending = [c for c in checks if c.passed is None]
    print("-" * 68)
    if failed:
        print(f"  {RED}BLOCKED{RESET} — {len(failed)} check(s) failing")
    elif pending:
        print(f"  {YELLOW}PENDING{RESET} — {len(pending)} check(s) not yet implemented")
    else:
        print(f"  {GREEN}GATE {number} PASSED{RESET} — advance to Phase {number + 1}")
    return not failed and not pending


def main() -> None:
    parser = argparse.ArgumentParser(description="GAFFER-RL phase gate checker")
    parser.add_argument("gate", nargs="?", type=int, choices=sorted(GATES))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        results = [run_gate(n) for n in sorted(GATES)]
        print(f"\n{'=' * 68}")
        print(f"  Gates passed: {sum(results)}/{len(results)}")
    elif args.gate is not None:
        sys.exit(0 if run_gate(args.gate) else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

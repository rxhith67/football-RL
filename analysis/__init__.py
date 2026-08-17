"""Evaluation harness, EGTA and figures. Owner C, Phases 1-3.

Planned modules:
  separability.py  K-way archetype confusion matrix — a GATE 1 condition, and it
                   runs on scripted archetypes long before any RL exists
  egta.py          payoff matrix, Nash (nashpy), alpha-Rank, and the scripted
                   non-transitivity check that is also a GATE 1 condition
  evaluate.py      match harness, bootstrap CIs, Elo, comeback rate
  figures.py       every plot shows the SEED SPREAD, not just the mean

`separability.py` and `egta.scripted_cycle_report` are wired into
`scripts/check_gate.py`; implementing them flips two PENDING checks on Gate 1.
"""

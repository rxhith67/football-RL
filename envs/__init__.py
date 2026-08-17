"""Gymnasium wrappers, belief filter and scripted opponents. Owner B, Phase 2.

Planned modules:
  coach_env.py   SMDP wrapper + action_masks() returning a FLAT length-20 array
  belief.py      Bayesian archetype filter (and the recurrent ablation)
  archetypes.py  scripted opponent coaches, WITH mandatory mid-match switching

Build against `pitchsim.base.PitchSim`, never against a concrete engine, so that
`StubPitchSim` can be swapped for the real one without touching this package.
"""

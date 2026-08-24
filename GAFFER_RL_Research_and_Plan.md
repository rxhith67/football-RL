# GAFFER-RL — Research Review & Implementation Plan

**Research review and implementation plan for tactical football RL**
Prepared 14 Aug 2026. Supersedes an earlier "DeepCoach-RL" draft.
Revised 15 Aug 2026 — phase-based plan, evidence gates, separability and non-transitivity made testable early, compute strategy.

---

## 0. Verdict in one page

**The *concept* in your Gemini draft is good. The *substrate* is broken.**

The good idea: stop doing frame-by-frame player control, and instead have an RL agent act as a **coach** that issues macro-tactical directives at a handful of decision points per match (a Semi-Markov Decision Process / options framing). That is genuinely under-explored, it collapses the credit-assignment horizon, it runs on a laptop, and it produces something a human can *watch and argue with* — which is exactly what a demo-and-review evaluation rewards.

The broken part: **Google Research Football (GRF) physically cannot execute that action space.** GRF's `step()` accepts one of 19 low-level player actions. It has no substitutions, no mid-match formation switching, no pressing parameter, and no xG. Roughly 60% of the draft's code would throw on first run, and the rest computes the wrong thing (details in §1). GRF is also Linux-only, needs a C++ build, and is effectively unmaintained.

**Recommendation: keep the coach idea, replace the substrate with your own environment, and add three things that make it novel rather than a re-skin.**

> ### GAFFER-RL
> *An RL football coach that infers the opposing manager's hidden game plan from the run of play, counter-punches with macro-tactical changes, and can be challenged live by a human coach.*
>
> **Three novelty pillars**
> 1. **Belief-conditioned coaching (POMDP).** The opponent's tactic is *hidden state*. The agent maintains an explicit, interpretable belief over opponent tactical archetypes, updated from observable match statistics (PPDA, field tilt, defensive line height, pass-length profile), and conditions its policy on that belief. Ablatable against a blind agent and an oracle agent.
> 2. **Empirical game-theoretic analysis of the tactic space.** Build the payoff matrix over tactical archetypes, compute Nash / α-Rank, and show that football's "rock–paper–scissors" structure exists in your sim and that the RL coach approximates a best-response *mixture* rather than a single dominant plan. This is the reportable scientific result.
> 3. **Human-in-the-loop adversarial demo.** The GUI lets a human be the *opposing manager* in real time, plus a counterfactual "what-if" panel that rewinds to minute *X*, forces a different tactic, and re-simulates to show the win-probability shift.
>
> **Why it is compelling:** it satisfies the assignment's "develop a custom Gym environment" route explicitly, the GUI is the point rather than a bolt-on, the compute fits Colab CPU, every claim is falsifiable, and review questions (SMDP discounting, potential-based shaping, self-play non-transitivity, POMDP belief states) all have crisp answers baked into the design.

If the sim build slips, there is a fully-specified fallback (§9, Option B: adversarial set-piece RL on StatsBomb 360 data) that reuses ~40% of the code. The plan is organised into five evidence-gated phases (§6), with the single real go/no-go at the end of the environment phase.

---

## 1. Audit of the DeepCoach-RL draft

I checked every technical claim in `football idea.txt` against the GRF source and docs. Findings, most severe first.

| # | Claim in draft | Reality | Severity |
|---|---|---|---|
| 1 | `self.grf_env.step(action)` with `MultiDiscrete([5,3,3,6])` applies formation/press/subs | GRF's action space is `Discrete(19)` per controlled player: idle, 8 move directions, long/high/short pass, shot, sprint, release direction, release sprint, slide, dribble, release dribble. **There is no team-level action of any kind.** | Fatal |
| 2 | Substitution actions `1..5` swap the most fatigued starter | **GRF has no substitutes.** There are no bench players in the engine, and no walkover when a team drops below 7 players. | Fatal |
| 3 | Formation is switchable mid-match | Formations exist only as starting positions/roles declared in a scenario file (`builder.AddPlayer(x, y, e_PlayerRole_CB)`), evaluated at kickoff. Not mutable at runtime without patching the C++ engine. | Fatal |
| 4 | `raw['steps_completed']`, 1500 steps = 90 min | The raw obs field is **`steps_left`**, and a full match is **`game_duration = 3000`** engine steps. The draft's 54,000 / 1500 / 83-step arithmetic is wrong three different ways; 5 match-minutes ≈ **167** steps. The headline "3,000× credit-assignment compression" is really ~167×. | High |
| 5 | `raw['right_team_tired_factor']` is our squad's stamina | The learner is the **left** team by default, so this reads the **opponent's** fatigue. | High |
| 6 | `tired_factor` is stamina, penalise when `< 0.35` | It is a **tiredness** factor: 0 = fresh, 1 = exhausted. The penalty term is inverted — it rewards exhausting your players. | High |
| 7 | ΔxG reward | **GRF has no xG.** `rewards="scoring"` gives ±1 on goals only. An xG model would have to be built from raw obs, and GRF exposes no shot-quality signal to fit it against. | High |
| 8 | `get_action_mask()` returns a list of 4 boolean arrays | `sb3-contrib`'s `MaskableMultiCategoricalDistribution` expects a **single flat boolean array of length `sum(nvec)`**, which it splits internally. A list will fail or silently mis-mask. *(Verify against the installed version — it has changed across releases.)* | Medium |
| 9 | Reward function in code = reward function in the maths | Code drops the terminal ±1, drops ΔxG entirely (`accumulated_reward` is raw goal reward), and applies the friction penalty on `any(action != last_action)` rather than the stated indicator. | Medium |
| 10 | "Fictitious self-play league" | The draft never controls the right-hand team, so checkpoints can never actually play each other. The opponent is always the fixed built-in bot. The league is decorative. | Medium |
| 11 | 100k timesteps ≈ 5,000 matches, converges in <4h on one GPU | 100k **macro**-steps ÷ ~18 per match ≈ 5,500 matches × 3,000 engine steps = **16.5M engine steps**. GRF runs a few hundred steps/s/core. That is days, not hours — and a 28-dim MLP policy does not need a GPU at all. | Medium |
| 12 | Install: `pip install gfootball` | Linux/Python 3.10 only, needs a full C++/SDL2/Boost build; Colab needs the unofficial prebuilt-`.so` + `GFOOTBALL_USE_PREBUILT_SO=1` hack; shipped checkpoints require TensorFlow 1.15. Windows is not supported. | Medium |
| 13 | ">70% win rate", "zero players below 20% stamina" | Unfalsifiable as written (no opponent specified, no sample size, no confidence interval) and #6 makes the stamina criterion meaningless. | Low |

**What survives and is worth keeping:** the SMDP/macro-decision framing, action masking for legality constraints, xG-based shaping as a variance reducer, self-play against a checkpoint pool, and the Streamlit dashboard concept. That is the skeleton of GAFFER-RL.

**One structural insight that drives the whole redesign:** to make macro-tactics mean anything in GRF you would have to control all 11 players and write your own low-level controller (formation-slot seeking, press triggers, pass selection). At that point *you have written a football AI anyway*, and GRF is contributing nothing but a 3D renderer and an install headache. So write the environment, and get a controllable, fast, inspectable, pip-free sim in exchange.

---

## 2. Research landscape (2019 → 2026)

### 2.1 What has been done

| Work | Year | What it does | Why it matters to you |
|---|---|---|---|
| **Google Research Football** (Kurach et al., arXiv:1907.11180) | 2019 | The canonical RL football env, 19 low-level actions | Your baseline citation for "why micro-control is hard" |
| **TiZero** (arXiv:2302.07515, AAMAS'23) | 2023 | Masters 11v11 GRF from scratch with curriculum + self-play | Cite as the **compute wall**: this is what full micro-control costs |
| **Empirical study of GRF multi-agent scenarios** (arXiv:2305.09458) | 2023 | Benchmarks MAPPO/QMIX etc. on GRF | Evidence that GRF results are hard to reproduce |
| **Game Plan: What AI can do for Football** (Tuyls et al., *JAIR* 71, arXiv:2011.09192) | 2021 | DeepMind + Liverpool FC agenda paper; game-theoretic analysis of penalty kicks with player-style vectors | **Direct precedent for your Pillar 2.** Cite this as the methodological anchor for EGTA in football |
| **TacticAI** (Wang et al., *Nature Communications*, 2024) | 2024 | GNN + geometric deep learning for corner-kick prediction & generation; Liverpool experts preferred its suggestions 90% of the time | The most famous football-AI result. **Not RL, and not open source** — a gap you can point at |
| **In-game soccer outcome prediction with offline RL** (*Machine Learning*, Springer) | 2024 | Builds an MDP from event data, offline RL to score actual-vs-optimal decisions | Precedent for MDPs fitted to real football data |
| **LCDSP — Complex Instruction Following with Diverse Style Policies** (arXiv:2511.19885, AAAI 2026) | 2025 | Single policy modulated by "style parameters"; a Style Interpreter maps natural-language tactical instructions to those parameters, in a 5v5 football env | Closest thing to "tell your team how to play". A natural **stretch goal** for you |
| **Optimising Football Corner Tactics with Graph RL** (arXiv:2606.06353) | Jun 2026 | Graph RL repositions attackers to maximise first-contact probability on 3,223 EPL corners; beats classical optimisation | Prior art for fallback Option B — and note it optimises against a **static** defence |
| **Hierarchical DRL for real-time tactical decision-making in team sports** (*Data Technologies and Applications*, 2026) | 2026 | Strategic/tactical/technical hierarchy, GNN + transformer, IRL opponent modelling | Closest prior art to GAFFER conceptually. Reported gains are implausibly clean; cite it, then differentiate on **reproducibility and an open environment** |

### 2.2 Where the gap actually is

Three observations after going through the above:

1. **Everyone works at one of two extremes.** Either micro-control of players in a physics engine (GRF, TiZero, LCDSP), or offline analytics on real data (TacticAI, offline-RL outcome models). The *manager's* layer — formation, block height, press intensity, tempo, substitutions, over the course of a match — is almost untouched as an RL control problem, despite being the layer humans actually argue about.
2. **Opponent tactics are treated as observed.** Self-play work conditions on raw state; analytics work conditions on data you already have. Nobody makes the agent *infer the opposing manager's plan from the run of play and act on that belief*. In real coaching this is the entire job.
3. **Nothing is both reproducible and watchable.** TacticAI: no code. GRF work: needs a cluster. The 2026 corner-kick RL paper: no released code. A small, open, fast environment with a coach's console is a real contribution at the project level, and it is precisely what the "Environment/GUI Design" and "Demo" deliverables reward.

That triple is your differentiation paragraph in the write-up, almost verbatim.

---

## 3. The project: GAFFER-RL

**Elevator pitch (for deliverable abstract):**
> Modern football-AI research either controls individual players in expensive physics simulators or analyses historical data offline; the manager's decision layer — shape, block height, press, tempo, substitutions — has been largely ignored as a control problem. GAFFER-RL formulates in-match coaching as a partially-observable Semi-Markov Decision Process over a fast, calibrated 2D football simulator. The agent makes ~18 macro-decisions per match, cannot see the opposing manager's game plan, and must infer it from observable match statistics. We train with maskable PPO under prioritised fictitious self-play, analyse the resulting tactic space with empirical game theory, and deliver a coach's console in which a human can manage the opposing side against the trained agent in real time.

### 3.1 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  GAFFER CONSOLE (browser)                                            │
│  broadcast pitch · belief bars · xG race · duel mode · what-if panel  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ replay JSON / live step
┌───────────────────────────▼──────────────────────────────────────────┐
│  COACH LAYER  —  SMDP, ~18 decisions per match                       │
│  ┌────────────────────┐   belief b_t   ┌───────────────────────────┐ │
│  │ Opponent-style      │──────────────▶│ MaskablePPO policy π(a|s,b)│ │
│  │ belief filter       │               │ MultiDiscrete[5,3,3,3,6]   │ │
│  └────────▲───────────┘               └────────────┬──────────────┘ │
│           │ observable match stats                 │ tactic vector    │
└───────────┼─────────────────────────────────────────┼────────────────┘
            │                                         ▼
┌───────────┴─────────────────────────────────────────────────────────┐
│  PLAYER LAYER  —  parameterised low-level controller (not learned)   │
│  formation slots · press triggers · pitch-control pass/shot choice   │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PITCHSIM  —  105×68 m, 22 players, 5 Hz, NumPy-vectorised           │
│  calibrated against StatsBomb open data                              │
└──────────────────────────────────────────────────────────────────────┘
```

The player layer is a **fixed, hand-written intra-option policy**. This is not a cop-out — it is exactly the options framework (Sutton, Precup & Singh 1999): the coach learns a policy *over* options whose internal policies are given. Say this explicitly in the write-up and review; it converts "we didn't train the players" from a weakness into a design decision with a citation.

### 3.2 PITCHSIM specification

The environment is the project. Freeze this spec in Phase 0 and do not extend it.

**Build it in two tiers.** Tier 1 (movement, passing, shooting, xG) is what the Phase-1 gate tests. Tier 2 (fouls/cards, set pieces, duel refinement) lands in Phase 2 and is explicitly *not* gated. Attempting both tiers before the first gate is the single most likely way this project dies — see §6.

**Physical layer** (all NumPy, batched over parallel matches; `dt = 1.0 s`, 90 min = 5,400 ticks)
- 105 × 68 m pitch, continuous `(x, y)` per player, per-player `max_speed` and `stamina ∈ [0,1]` (1 = fresh).
- Movement: each player steers toward a **target point** = formation slot, translated by ball position (compactness) and modulated by block height and team width. Speed scales with stamina; stamina drains faster under high press and sprinting, recovers when out of possession phases.
- Ball: constant-deceleration model, position + velocity, plus a flight time for aerial passes.

**Event layer** (fires only when a player is in possession, every 2.5 s on a shared batched clock)
- Candidate actions: short pass, long/switch pass, dribble/carry, shot, clearance.
- Selection: softmax over an **expected-value score** for each option = `pitch_control(target) × EPV(target)` minus a risk term, with the softmax temperature and the risk term set by the coach's *tempo/directness* parameter. Pitch control uses a simplified Spearman-style time-to-intercept model — the Friends-of-Tracking `Metrica_PitchControl.py` reference implementation is the pattern to follow (do not copy wholesale; re-implement in vectorised form).
- Defensive side: press triggers fire when the ball enters a team's press zone (set by block height + press intensity); duel outcomes are stochastic with probabilities depending on numbers-up/down locally and stamina.
- Shots: xG from a logistic model on distance + angle + defenders-in-cone, **fitted on StatsBomb open-data shots**. Goal sampled from xG. This gives you a *real*, defensible xG rather than the draft's placeholder.
- Set pieces: corners, throw-ins, free kicks handled as simplified restarts (fixed xG lottery + restart possession). Do not over-model these; they are Option B's territory.
- Fouls/cards: Poisson rate scaling with press intensity — this is what makes gegenpress genuinely risky.

**Calibration targets** (deliverable deliverable — a table of sim vs real)

Real-world values below are **measured**, not quoted from memory: 8,536 shots
across 350 matches of StatsBomb open data (La Liga, Premier League, Bundesliga,
Euro 2020, WC 2022, WWC 2023, WEuro 2022), extracted 15 Aug 2026. Regenerate with
`python -m pitchsim.statsbomb --summary`.

| Metric | Real (measured) | Sim acceptance band |
|---|---|---|
| Goals / match (both teams) | **2.49** | 2.3 – 3.1 |
| Shots / match / team | **12.19** | 9 – 15 |
| xG per shot | **0.0985** | 0.08 – 0.13 |
| Pass completion % | ≈ 80% *(not yet measured — the shot table carries no passes; measure in Phase 1)* | 72 – 88% |
| Possession share vs tactic | monotone in tempo/press | must respond in the right direction |
| Home/neutral win-draw-loss split | ≈ 45/25/30 (neutral: symmetric) | symmetric within noise |
| **Archetype separability** (see below) | n/a — a property of the sim | ≥ 70% 6-way accuracy from a 15-min window |

**Archetype separability is a first-class calibration target, not an afterthought.** The parameters that make the sim *realistic* are the same ones that make tactics *distinguishable*, and they pull against each other: a sim tuned only for 2.7 goals/match can easily end up one where every archetype produces the same PPDA and field tilt, at which point the belief filter (Pillar 1) has nothing to read and the whole first pillar collapses. So tune both jointly, and measure separability with a concrete number: fit the per-archetype Gaussians of §3.4, classify held-out 15-minute windows, and require ≥ 70% MAP accuracy on the K-way confusion matrix — **with no archetype below 50% recall** (one indistinguishable pair is enough to hollow out the ablation). Produce this confusion matrix in Phase 1, *before* any RL code touches the belief vector.

Publishing this table is valuable than any amount of reward-curve polish: it is the evidence that your environment models a *real-world problem context*, which is the stated intent of the evaluation.

**Performance target:** a sustained ≥ 60 simulated matches/minute on the presenting laptop at 48 vectorised environments. The cold rate is roughly 98; the lower gate is the post-load thermal floor measured while running the complete audit. Horizontal scale-out remains the lever for final evaluation and sweeps.

### 3.3 The MDP (state this precisely in the write-up)

**Decision epochs.** The coach acts at `t = 5, 10, …, 85` sim-minutes, **plus** on triggers: goal scored/conceded, red card, half-time, and "opponent tactic-shift detected" (belief entropy jump). This makes durations `τ_k` variable ⇒ genuine SMDP, so discount with `γ^{τ_k}` rather than a flat `γ`. Expect ~18–24 decisions per match.

**Action space** `MultiDiscrete([5, 3, 3, 3, 6])` = 810 combinations:

| Head | Values | Effect in PITCHSIM |
|---|---|---|
| Formation | 4-3-3, 4-2-3-1, 3-5-2, 5-4-1, 4-4-2 | swaps the slot-coordinate matrix |
| Block height | low / mid / high | shifts the defensive line's rest y-position and offside trap depth |
| Press intensity | contain / balanced / gegenpress | press-trigger radius, closing speed, stamina drain, foul rate |
| Tempo | patient / balanced / direct | softmax temperature + risk weight in the pass-selection score |
| Substitution | none / bench 1–5 | restores stamina at that slot; costs a sub and a window |

**Action masking** (the honest use of `MaskablePPO`, unlike the draft): subs unavailable once 5 are used or 3 windows are spent; a back-three formation masked out if fewer than 3 fit centre-backs remain; no substitution in the first 15 minutes (a soft realism constraint); tactic-change lockout for one epoch after a change. Return a **flat boolean array of length 20** (`5+3+3+3+6`).

**Observation** (≈ 55 dims, all normalised):
- Match state (5): score diff, minutes remaining, subs left, windows left, own red-card flag.
- Squad (13): 11 stamina values, count below threshold, yellow-card count.
- Own rolling form over the last window (9): possession, shots for, defensive-line height, xG for/against, field tilt, PPDA, pass completion and mean pass length.
- Opponent observables (9): the same statistics from the opponent's side.
- Own current tactic, one-hot (14).
- **Belief over opponent archetype `b_t` (K ≈ 6–8).**

**Reward.**
```
R_k = R_terminal·1[k = final]  +  Φ(s_{k+1}) − Φ(s_k)  −  λ_fatigue·f_k  −  λ_churn·c_k
```
- `R_terminal = +1 / 0 / −1` for win/draw/loss.
- `Φ(s) = w·(xG_for − xG_against)` — a **potential-based shaping function** (Ng, Harada & Russell 1999). This is a real improvement over the draft's ad-hoc `α·ΔxG`: potential-based shaping is provably policy-invariant, so you get the variance reduction without changing the optimal policy. It is also a guaranteed review question you will nail.
- `f_k` = mean squared stamina deficit below 0.35 (**correctly signed**, unlike the draft), `λ_fatigue ≈ 0.3`.
- `c_k` = 1 if any tactic head changed, `λ_churn ≈ 0.02` — keeps the policy legible for the demo.

**Reward scale — set `w` deliberately, and justify it in the write-up.** Φ telescopes, so over a full match the shaping term sums to `w·(xG_for − xG_against)|_final` regardless of path; with ~1.5 xG per team per match, a typical match-end xG difference is ≈ ±1.0 and a large one ≈ ±2.5. Choose `w` so that total shaping magnitude sits *below* the terminal reward: `w ≈ 0.3` gives typical shaping ≈ 0.3 against a terminal ±1. Too large and the agent optimises xG and shrugs at the scoreline (it will happily lose 0–1 while winning the xG battle); too small and shaping does nothing and you have merely added a hyperparameter. Report the sensitivity — `w ∈ {0, 0.15, 0.3, 0.6}` is a cheap, genuinely informative ablation, and "shaping did not help, here is the CI" is a perfectly good result.

**Partial observability.** The opponent's tactic vector is *never* in the observation. Only its statistical footprint is.

### 3.4 The belief filter (Pillar 1)

Two implementations, both cheap; do the first, report the second as an ablation:

1. **Bayesian filter over archetypes.** Define K opponent archetypes (e.g. *high-press 4-3-3*, *low-block 5-4-1*, *counter 4-4-2*, *possession 4-2-3-1*, *direct 3-5-2*, *balanced*). Offline, simulate a few thousand windows per archetype and fit per-archetype Gaussians over the 8-dim observable-statistics vector `o`. Online: `b_{k+1} ∝ b_k · N(o_k; μ_j, Σ_j)`, with a small transition matrix allowing the opponent to switch. Fully interpretable — and the belief bars in the GUI are *literally* this vector.
2. **Learned recurrent belief.** `RecurrentPPO` (LSTM) with no explicit belief input, letting the hidden state do the work. Compare final Elo and sample efficiency against (1).

**Scripted opponents must switch tactics mid-match. This is not optional.** If the opponent picks an archetype at kickoff and holds it for 90 minutes, the filter saturates to near-certainty within ten minutes and *belief ≈ oracle* — the ablation table below flattens and Pillar 1 has no result to report. The failure is symmetric: if archetypes are indistinguishable (§3.2) then *blind ≈ belief*, and the table flattens the other way. Pillar 1 only produces a finding in the band between those two degeneracies, so engineer the band rather than hoping for it:

- Every scripted archetype opponent switches on its own schedule — a state-dependent rule (trailing at half-time ⇒ raise press and go direct; leading after minute 70 ⇒ drop the block), plus a small random switch hazard. This is also just more realistic; a manager who never reacts is not a manager.
- The belief transition matrix in filter (1) must therefore be genuinely non-diagonal, and its off-diagonal mass is a hyperparameter worth reporting.
- **This makes adaptation latency (§5, metric 4) a real metric**: not "how long to first identify" but "how long to *re-*identify after a switch", measured from the true switch epoch. That is the number the belief bars in the console visibly dramatise, and it is the one an readers will remember.

**The ablation table is the paper-shaped result of the project:**

| Agent | Opponent info | Expected outcome |
|---|---|---|
| Blind | none | floor |
| Belief (Bayesian) | `b_t` | should recover most of the oracle gap |
| Recurrent | implicit | comparable, less interpretable |
| **Oracle** | true opponent tactic one-hot | ceiling |

Report Elo, win rate vs the fixed pool, and points-per-match with 95% CIs over ≥ 500 matches per cell.

### 3.5 Training curriculum

- **Stage 0 — sanity.** Fixed opponent (balanced 4-4-2, mid block). Learner must beat a random-tactic coach and a "never change anything" coach. If it doesn't, the environment is broken, not the agent.
- **Stage 1 — archetype gauntlet.** Uniform random opponent from the K scripted archetypes. This is where the belief filter starts paying.
- **Stage 2 — prioritised fictitious self-play.** Both sides are coach policies. Maintain a checkpoint pool; sample opponents with probability weighted by `(1 − win_rate)` (PFSP, as in AlphaStar). Snapshot every N updates. **This is real self-play** — symmetric env, both teams coached — unlike the draft.
- **Stage 3 — EGTA (Pillar 2).** Freeze a pool = {K archetypes + 4–6 RL checkpoints}. Play a round-robin, ≥ 300 matches per pairing, build the payoff matrix. Compute the Nash equilibrium with `nashpy` (2-player zero-sum ⇒ an LP) and/or α-Rank. Report: (a) is the tactic space non-transitive (does an RPS cycle exist)? (b) does the final agent's *empirical* tactic-usage distribution resemble the Nash mixture? (c) does adding RL checkpoints to the pool change the equilibrium support?

**Verify the non-transitive cycle exists in Phase 1 — with scripted archetypes only, no RL involved.** Whether football's rock–paper–scissors structure appears in your payoff matrix is a property of *your event layer's EV score*, not a law of nature. If the low-level controller makes gegenpress strictly dominant, the matrix is transitive, α-Rank returns a single-support answer, and Pillar 2's headline result evaporates in Phase 3 with no time left to fix it. Discovering that in the final fortnight is the worst outcome in this plan.

So: as soon as the K archetypes run (Phase 1, immediately after the separability confusion matrix — it is the same round-robin harness), play a K×K scripted round-robin at ~200 matches per cell and inspect the matrix for a cycle. The counters should be *designed into* the EV score of §3.2, not left to emerge:

| Beats | …because |
|---|---|
| High press → patient possession | press triggers force turnovers high up against short build-up |
| Patient possession → low block | sustained territory eventually finds the gap; low block cedes the ball |
| Direct/long-ball → high press | direct balls bypass the press line entirely; the high line is exposed in behind |
| Low block → direct/long-ball | numbers behind the ball absorb aerial route-one |

That is two interlocking cycles, and it is exactly how football people describe the game — which makes it defensible rather than contrived. If the Phase-1 round-robin shows no cycle, retune the EV terms until one appears, and **document the tuning honestly in the write-up**: "we designed the environment to admit non-transitivity, then asked whether the RL coach discovers and exploits it" is a legitimate and clearly-stated experimental question. Claiming to have *discovered* a cycle you engineered is not — do not let the write-up drift into that.

**Algorithms to run** (the comparison the evaluation criteria wants):

| Algorithm | Role | Library |
|---|---|---|
| Random tactic / static tactic / greedy heuristic | baselines | your own |
| Tabular Q-learning over a coarsely discretised state (spec below) | shows you understand the theory; will lose, and *why* it loses is a great report paragraph | your own |
| DQN | value-based deep baseline | SB3 (flatten the MultiDiscrete or use one head) |
| **MaskablePPO** | main method | `sb3-contrib` |
| RecurrentPPO | belief ablation | `sb3-contrib` |

**Tabular Q-learning — specify the discretisation or it is not an experiment.** A Q-table over the ~55-dim continuous observation is not merely impractical, it is undefined; "we tried tabular and it was bad" earns nothing. Commit to this state abstraction and state it in the write-up:

- score diff ∈ {−2 or worse, −1, 0, +1, +2 or better} (5)
- match phase ∈ {0–30, 30–60, 60–90 min} (3)
- own mean stamina ∈ {low, mid, high} (3)
- MAP opponent archetype (K = 6)
- own current tactic, coarsened to {formation, press} only (5 × 3 = 15)

⇒ 5·3·3·6·15 = **4,050 states**, against 810 raw actions — so also coarsen the action space for this baseline to {formation × press} = 15, giving ~61k Q-values. That is learnable within your match budget, and the resulting report paragraph writes itself: the abstraction discards tempo, fatigue detail and belief *uncertainty* (it keeps only the MAP, throwing away the entropy that drives the deep agent's hedging), so it plateaus below MaskablePPO — a concrete, quantified demonstration of why function approximation matters, which is precisely what the criteria is looking for.

### 3.6 Stretch goals (only if Phase 3 opens on schedule)

- **Natural-language coaching** (LCDSP-flavoured): map free text — "sit deep and hit them on the break" — onto the tactic vector via a small classifier or a local LLM call, and let the demo audience *type* instructions. Very high demo value, low technical risk since the tactic space is tiny and discrete.
- **Squad heterogeneity:** per-player attributes (pace, stamina, finishing) sampled per match, so substitution decisions become genuinely non-trivial.
- **Injury/red-card shocks** as an adaptation stress test.

---

## 4. The GAFFER Console (deliverable — GUI design)

Two-process design, chosen so real-time animation is not fighting Streamlit's rerun model:

- **Backend:** FastAPI. Endpoints: `POST /simulate` (returns a full replay JSON: per-tick positions, events, tactic changes, belief trace, win-prob trace), `POST /step` (duel mode: advance one 5-minute block given the human's tactic), `POST /whatif` (rewind to minute *X*, force tactic *a*, run N rollouts, return the win-probability distribution).
- **Frontend:** a single HTML page with a `<canvas>` pitch renderer + Chart.js panels. No build step, no npm, works offline, embeds cleanly in a slide deck.

*(If the team is uncomfortable with JS: Streamlit + a pre-computed replay and a minute slider is an acceptable downgrade. Scrubbing a pre-computed replay is actually a **better** demo than live animation — you can jump to the interesting moment on stage. Make this call at the Phase 1 → Phase 2 boundary, while there is still time to act on it — not in the final fortnight.)*

**Panels:**

| Panel | Content | Why it adds value |
|---|---|---|
| **Broadcast pitch** | 22 dots + ball, formation-shape overlay, toggleable pitch-control heatmap, press-zone shading | "operational use of the algorithm" — the readers sees the tactic, not just a number |
| **Manager's bench** | The agent's chosen tactic per epoch, with the top-3 state features driving it and the value-function delta vs the next-best action | explainability; answers "why did it do that?" in review before it's asked |
| **Opponent read** | Live belief bars over the K archetypes + a confidence trace; flag the moment the read flips | this *is* Pillar 1, made visible |
| **xG race & win probability** | cumulative xG for/against, live win-prob from the critic | the classic football-broadcast graphic; instantly legible |
| **Squad bar** | stamina per player, sub windows remaining, cards | makes the substitution policy watchable |
| **⚔️ Duel mode** | Human picks the opposing tactic each 5 minutes; the agent responds; scoreboard across matches | **the demo moment** — hand the laptop to the tutor |
| **🔁 What-if** | Rewind to minute *X*, force a tactic, re-simulate 200×, show the win-prob shift as a distribution | counterfactual reasoning; the TacticAI-flavoured "wow" panel |

**Presentation script (5 min):** show one full match replay with the belief flipping mid-match and the agent counter-punching → invite the tutor to coach against it in duel mode → close with the what-if panel proving the agent's minute-60 decision was worth +X% win probability → one slide with the EGTA payoff matrix and the RPS cycle.

---

## 5. Evaluation protocol

Fix this **before** training and pre-register it. Sloppy evaluation is the single most common way a simulation study misleads itself.

- **Opponent pool:** 6–8 scripted archetypes + 3 RL checkpoints + 1 random-tactic coach.
- **Sample size:** ≥ 500 matches per (agent, opponent) cell; report win/draw/loss, points-per-match, goal difference, xG difference, all with 95% bootstrap CIs.
- **Seeds:** ≥ 5 training seeds per configuration. **Plot the seed spread, not just the mean** — this alone puts you ahead of most submissions.
- **Headline metrics:**
  1. Elo of each agent within the pool.
  2. Win rate vs the fixed archetype pool (Stage-1 generalisation).
  3. **Comeback rate:** P(win or draw | trailing at half-time) — the metric a football person actually cares about, and the one your macro-coach should move.
  4. Adaptation latency: epochs until the belief filter's MAP archetype is correct.
  5. Exploitability proxy: train a fresh best-response coach against the frozen final agent for a **fixed, stated budget** — 200k macro-steps, identical PPO hyperparameters, 3 seeds — then report its points-per-match over 500 evaluation matches. Without a declared budget the number means nothing (any agent is exploitable given unbounded compute), so the budget belongs in the caption, not just the appendix. Lower is more robust; this is the honest version of the draft's self-play claim. Run the same probe against a *scripted* archetype as a reference point, so the reader has a scale to judge against.
- **Ablations:** belief vs blind vs oracle vs recurrent; potential-based shaping on/off; action masking on/off; SMDP `γ^τ` vs flat `γ`.
- **Negative results are fine and should be reported.** If shaping doesn't help, say so with the CI. readers reward this in oral defence.

---

## 6. Phased plan (14 Aug → 23 Oct 2026)

### 6.1 Ownership

Three people, independently owned, so ownership boundaries matter.

- **A — Environment owner:** PITCHSIM physics/event layer, calibration, low-level controller, archetype scripts.
- **B — Learning owner:** Gym wrapper, MDP, masking, PPO/DQN/tabular runs, belief filter, self-play league, EGTA.
- **C — Interface & evidence owner:** FastAPI backend, console, replay format, evaluation harness, compute infrastructure, figures, report assembly.

Everyone should understand the *whole* system in review — hold a standing 30-minute cross-explain each week where each person teaches their part to the others.

**Break the serial dependency on day one.** As written, the learning and interface layers are blocked on the environment for the first third of the project. The fix is cheap and non-negotiable: **the first environment deliverable is a stub PITCHSIM** — the real API signatures (`reset`, `step`, the observation vector, the replay JSON schema) returning randomised outcomes with plausible statistics. The Gym wrapper, masking and baselines build against the stub, as do the replay renderer and evaluation harness. When the real engine lands it drops in behind an unchanged interface. Freeze the replay JSON schema at the same time and version it; it is the contract between all three layers.

### 6.2 The phases

Phases are gated by *evidence*, not by dates. The calendar anchors below are targets; the gates are what actually decide whether you advance. Each phase names the one thing that kills the project if skipped.

---

**Phase 0 — Foundations & de-risking** · target 14 Aug → 27 Aug · *owner: all three*

Freeze the §3.2 spec. Repo skeleton, ownership written into the README, StatsBomb loader, xG model fitted on real shots, stub PITCHSIM with frozen interfaces, deliverable abstract drafted from §3.

> **Gate 0:** xG model AUC > 0.75 on held-out shots · stub PITCHSIM importable and B/C both building against it.

This phase is nearly risk-free and useful under *both* GAFFER and Option B, which is exactly why it goes first.

---

**Phase 1 — The environment** · target 28 Aug → 17 Sep · *owner: A, with B on archetypes*

The long pole, and the phase where honest scheduling matters most. A vectorised 22-agent physics loop with formation-slot steering, press triggers, pitch control, EV-based pass selection and a calibration pass is **three to four weeks of real work**, not one. Budget it as such. Deliver in two tiers:

- **Tier 1 (gated):** movement + formation slots, ball model, pass/dribble/shot selection, xG-sampled goals. Matplotlib animation of one match.
- **Tier 2 (not gated, lands during Phase 2):** fouls/cards, set pieces, duel refinement, stamina tuning.

Then, still in Phase 1: vectorise to 32 envs and profile; run the §3.2 calibration pass; script the K archetypes and run the K×K round-robin that produces **both** the separability confusion matrix (§3.2) and the non-transitivity check (§3.5). These two artefacts decide whether Pillars 1 and 2 have anything to report, and they cost one harness and a few CPU-hours — finding out now is the entire point.

> **HARD GATE 1 — the project's one real go/no-go.** All four must hold:
> 1. **A match that looks like football** — a football-literate person watching the animation says yes, unprompted.
> 2. Calibration bands in §3.2 met, and the sustained local throughput floor in §3.2 cleared.
> 3. Archetype separability ≥ 70% with no archetype below 50% recall.
> 4. A non-transitive cycle exists in the scripted payoff matrix (after deliberate EV tuning, per §3.5).
>
> **If (1) fails, switch to Option B immediately and without debate.** If (2)–(4) fail, they are tuning problems: fix them inside Phase 1 rather than advancing, because every one of them silently voids a downstream deliverable.

**deliverable falls due during this phase** — write it from the frozen spec and the stub-based baselines; it does not depend on the real engine being finished.

---

**Phase 2 — Learning** · target 18 Sep → 8 Oct · *owner: B, with C on console v1*

Gymnasium SMDP wrapper, `γ^τ` discounting, flat-array action masking, the §3.3 reward with `w` set per the scale argument there. Baselines (random / static / greedy / tabular Q at the §3.5 discretisation). Then MaskablePPO through Stage 0 → Stage 1 → Stage 2 PFSP self-play with Elo tracking. Belief filter wired in, with the blind/belief/oracle/recurrent ablations. C ships console v1 (replay scrubbing + broadcast pitch) and the evaluation harness.

> **Gate 2:** PPO beats the random-tactic coach with non-overlapping 95% CIs · belief MAP re-identifies a mid-match tactic switch within 3 epochs · Elo rises across self-play generations · console renders a real replay end to end.

**Parts C and D fall due during this phase.**

---

**Phase 3 — The science** · target 9 Oct → 15 Oct · *owner: B on analysis, C on console*

EGTA proper: freeze the pool, round-robin, payoff matrix, Nash via `nashpy` and α-Rank; answer the three questions in §3.5 Stage 3. Exploitability probe at the declared budget. All remaining ablations (shaping on/off and the `w` sweep, masking on/off, `γ^τ` vs flat `γ`). Console duel mode and the what-if panel. Every figure final.

> **Gate 3 — FEATURE FREEZE, 15 Oct, no exceptions.** Every number that will appear in the write-up exists on disk. Nothing new is built after this date.

Stretch goals (§3.6) are considered *only* if this phase opens on schedule — and abandoned the moment Gate 3 is at risk.

---

**Phase 4 — Consolidation** · target 16 Oct → 23 Oct · *owner: C leads, all three write*

Report assembly, demo rehearsal ×3 on the actual presenting laptop, review Q&A bank (SMDP discounting, potential-based shaping, non-transitivity, POMDP belief states — see §3 and §5).

> **Gate 4:** the full demo runs offline, from cached replays, on the presenting laptop, with the network cable pulled. **milestone 23 Oct.**

**Buffer discipline:** Phase 4 is for writing and rehearsal, not coding. If you are still training in Phase 4, the demo will fail on stage. This is the most reliably-predicted failure in student projects and the easiest to avoid.

### 6.3 Slip protocol

Phases slip; that is expected and survivable if you know in advance what you will cut. Cut in this order, and decide at the gate rather than drifting:

1. Stretch goals (§3.6) — first out, always.
2. Recurrent-belief ablation — the Bayesian filter alone carries Pillar 1.
3. What-if panel — duel mode is the demo moment; what-if is the garnish.
4. DQN baseline — tabular Q plus the scripted baselines already demonstrate the algorithmic comparison.
5. Self-play Stage 2 — Stage 1's archetype gauntlet plus EGTA over scripted archetypes still supports both pillars.

**Never cut:** the calibration table, the blind/belief/oracle ablation, seed spread on every plot, or the offline demo rehearsal. Those four are where the credibility of the whole thing lives.

### 6.4 Compute — and why you probably do not want an A100

**Your bottleneck is CPU, not GPU.** PITCHSIM is a NumPy physics loop; the policy is an MLP over ~55 inputs. On an A100 the GPU idles near zero utilisation while a single core does all the work, and you pay A100 rates for the privilege. §1 (row 11) already makes this point about the draft's compute claim, and it applies just as much to the final design.

What is genuinely worth renting is **horizontal CPU fan-out**, and Modal is a good fit for it:

| Workload | Shape | Why it wants fan-out |
|---|---|---|
| EGTA round-robin | ~10 × 10 pairings × 300 matches ≈ 30k matches | Embarrassingly parallel; one container per pairing turns hours into minutes |
| Evaluation cells | ≥ 500 matches × every (agent, opponent) pair | Same |
| Seed sweeps | ≥ 5 seeds × several configs | Independent runs; the seed spread §5 demands is free once parallel |
| `w` / ablation sweeps | 4 shaping values × seeds | Same |

Practically: define CPU-only Modal functions (no `gpu=` argument at all), keep checkpoints and replay JSON on a Modal Volume, and push only *tagged* checkpoints and final artefacts to a Hugging Face repo for durable backup via `huggingface_hub`. HF Hub is built for models and datasets and is free at your scale, but it is not a general blob store — do not sync raw logs and per-tick replays there. If the repo is public, honour StatsBomb's open-data attribution requirements.

**The two cases where a GPU actually earns its cost:**

1. **A JAX/Brax rewrite of PITCHSIM**, running thousands of matches on-device. This is a real and attractive option — but it is a functional-style rewrite of the entire engine, and given that Phase 1 is already the long pole, it is out of scope. If you want it, commit in Phase 0 or not at all.
2. **The natural-language coaching stretch goal** (§3.6), if you run a local LLM. Even there an L4 or T4 is ample; a small instruct model mapping free text onto a 5-head discrete tactic vector is not an A100 workload.

So: rent CPU breadth, keep a modest GPU in your pocket for the stretch goal, and do not let "we had A100s" substitute for the thing that actually determines the grade, which is a calibrated environment and a clean ablation table.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Phase 1 overruns** (the environment is 3–4 weeks of work, not 1) | **High** | High | Two-tier build: only Tier 1 is gated. Stub PITCHSIM keeps B and C productive throughout. Slip protocol §6.3 decided in advance |
| **The sim doesn't look like football** | Medium | Fatal | Gate 1 with an explicit fallback (Option B), triggered on evidence not on a date. Calibrate against real data, don't eyeball |
| Sim too slow in pure Python | Medium | High | Vectorise across matches from day one; escalation ladder in §3.2 (`dt` → match length → Numba); CPU fan-out per §6.4 for evaluation |
| Self-play collapses to one degenerate tactic | Medium | Medium | PFSP sampling, keep archetypes permanently in the pool, entropy bonus, report the collapse if it happens (it's a finding) |
| Scope creep into player-level learning | High | High | Player layer is **fixed by design** (options framework). Written into the spec. Refuse it |
| Belief filter learns nothing (archetypes statistically indistinguishable) | Medium | Medium | Separability is a numbered calibration target (§3.2), verified by confusion matrix at Gate 1 *before* wiring to RL |
| **Belief filter is trivially easy ⇒ belief ≈ oracle, ablation flattens** | Medium | Medium | Scripted opponents switch tactics mid-match by rule + hazard (§3.4); adaptation latency measured from the true switch epoch |
| **Tactic space is transitive ⇒ Pillar 2 has no result** | Medium | High | Counters designed into the EV score; cycle verified on scripted archetypes at Gate 1, not discovered in Phase 3 |
| Shaping term swamps or ignores the terminal reward | Medium | Medium | `w ≈ 0.3` from the scale argument in §3.3; report the `w ∈ {0, 0.15, 0.3, 0.6}` sweep either way |
| Uneven contribution (independently owned) | Medium | High | Hard ownership split §6.1, stub-first to break the serial dependency, per-person commit history, weekly cross-explain |
| Compute spend on the wrong resource (GPU idle while CPU bottlenecks) | Medium | Low | CPU-only Modal functions; GPU reserved for the stretch goal alone (§6.4) |
| Demo fails live | Medium | High | Pre-computed replays cached to disk; demo runs fully offline; rehearse 3× on the actual presentation laptop |
| Accusation of "just a simulator, not RL" | Low | Medium | Lead every document with the MDP formalism, the SMDP discounting, the ablation table, and the algorithm comparison |

---

## 8. Repository skeleton

```
gaffer_rl/
├── README.md
├── requirements.txt                 # gymnasium, sb3, sb3-contrib, numpy, scipy,
│                                    # pandas, scikit-learn, nashpy, fastapi, uvicorn
├── config/
│   ├── pitchsim.yaml                # physics constants, calibration parameters
│   ├── tactics.yaml                 # formation slot matrices, archetype definitions
│   └── train.yaml                   # PPO hyperparameters, curriculum stages
├── data/
│   └── statsbomb/                   # cached open-data shots for the xG fit
├── pitchsim/
│   ├── engine.py                    # vectorised physics + event loop
│   ├── controller.py                # parameterised low-level player policy
│   ├── pitch_control.py             # time-to-intercept surface
│   ├── xg.py                        # logistic xG fitted on StatsBomb
│   └── calibrate.py                 # sim-vs-real report generator
├── envs/
│   ├── coach_env.py                 # Gymnasium SMDP wrapper + action_masks()
│   ├── belief.py                    # Bayesian archetype filter
│   └── archetypes.py                # scripted opponent coaches
├── agents/
│   ├── baselines.py                 # random / static / greedy / tabular Q
│   └── league.py                    # PFSP checkpoint pool + Elo
├── analysis/
│   ├── egta.py                      # payoff matrix, Nash, alpha-rank
│   ├── evaluate.py                  # match harness, bootstrap CIs
│   └── figures.py
├── console/
│   ├── server.py                    # FastAPI: /simulate /step /whatif
│   └── static/                      # index.html, pitch.js, charts.js
├── train.py
└── notebooks/
    └── AT3_demo.ipynb               # Colab-runnable end-to-end notebook (release format)
```

---

## 9. Alternatives, if you want a different risk profile

### Option B — Set-Piece Lab *(the fallback; also a fine primary if you prefer real data over simulation)*

**Pitch:** an adversarial RL corner-kick designer. Attacking and defending coaches both learn, via self-play, to position and move players during a corner; evaluated on real StatsBomb 360 freeze-frames.

- **Data:** StatsBomb open data (free) — event data plus 360 freeze-frames giving all visible player positions at each event, across Euros, World Cups, FA WSL and selected league seasons.
- **Model:** fit a first-contact / shot-probability model on real corners (GNN or gradient boosting on relational features), then run RL where actions are player movements over the ~4 seconds of the delivery, with a kinematic time-to-intercept model providing the dynamics.
- **Novelty vs prior art:** TacticAI (2024) is generative, not RL, and closed-source. arXiv:2606.06353 (Jun 2026) is RL but optimises the attack against a **static defence**. **A two-sided, zero-sum attack-vs-defence game with an equilibrium routine is the open gap** — and it's the natural RL framing.
- **UI:** drag players on a pitch, press *Optimise*, watch the routine resolve; toggle "defence responds"; show the threat surface.
- **Pros:** real data, tiny compute, gorgeous demo, strong citations. **Cons:** thinner data than you'd like; risk of the RL agent reward-hacking the learned outcome model (mitigate by holding out corners chronologically and validating on unseen ones — and report the hacking if it happens, it's interesting).
- **Phase mapping if you fall back at Gate 1:** Phase 0 carries over almost entirely (StatsBomb loader, xG model, repo, ownership). Phase 1 becomes "fit the first-contact model on real corners"; Phases 2–4 keep their shape and their gates. This is why Phase 0 is deliberately option-agnostic — falling back costs you the environment work and nothing else.

### Option C — Corrected GRF *(only if someone insists on Google Research Football)*

Minimum viable honest version: use a `5_vs_5` or academy scenario, control all 5 players (`number_of_left_players_agent_controls=5`), and treat *your own* scripted formation/press controller as the low-level layer, with the RL coach selecting among 4–6 macro-styles. Accept that substitutions are impossible, formations are set only at kickoff, and xG must be built by you. Budget one full week purely for installation (Colab prebuilt-`.so` route) before committing. **I do not recommend this** — you'd spend the week on a build system and end up writing the same low-level controller anyway, but with a 3D renderer as the only payoff.

---

## 10. References for the write-up bibliography

| # | Reference | Use |
|---|---|---|
| 1 | Kurach et al., *Google Research Football: A Novel RL Environment*, arXiv:1907.11180 (2019) | Environment landscape; why micro-control is expensive |
| 2 | Tuyls et al., *Game Plan: What AI can do for Football, and What Football can do for AI*, JAIR 71 (2021), arXiv:2011.09192 | **EGTA in football — anchor for Pillar 2** |
| 3 | Wang et al., *TacticAI: an AI assistant for football tactics*, Nature Communications (2024) | State of the art in football AI; gap: no RL, no code |
| 4 | Lin et al., *TiZero: Mastering Multi-Agent Football with Curriculum Learning and Self-Play*, arXiv:2302.07515 (AAMAS 2023) | The compute wall for micro-control |
| 5 | *An Empirical Study on Google Research Football Multi-agent Scenarios*, arXiv:2305.09458 (2023) | Reproducibility of GRF results |
| 6 | *Complex Instruction Following with Diverse Style Policies in Football Games*, arXiv:2511.19885 (AAAI 2026) | Language-conditioned tactics; stretch goal |
| 7 | *Maximising the Set-Piece Return: Optimising Football Corner Tactics with Graph RL*, arXiv:2606.06353 (2026) | Option B prior art; static-defence limitation |
| 8 | *Hierarchical DRL for real-time tactical decision-making in team sports*, Data Technologies and Applications 60(2) (2026) | Closest conceptual prior art |
| 9 | *In-game soccer outcome prediction with offline RL*, Machine Learning (Springer, 2024) | MDPs fitted to real football data |
| 10 | Sutton, Precup & Singh, *Between MDPs and semi-MDPs: A framework for temporal abstraction*, AIJ 112 (1999) | **SMDP / options — the formal basis of the coach layer** |
| 11 | Ng, Harada & Russell, *Policy invariance under reward transformations*, ICML (1999) | **Potential-based shaping — why our ΔxG term is safe** |
| 12 | Vinyals et al., *Grandmaster level in StarCraft II* (AlphaStar), Nature 575 (2019) | PFSP league training |
| 13 | Lanctot et al., *A Unified Game-Theoretic Approach to MARL* (PSRO), NeurIPS (2017) | League/EGTA methodology |
| 14 | Omidshafiei et al., *α-Rank: Multi-Agent Evaluation by Evolution*, Scientific Reports (2019) | Ranking in non-transitive games |
| 15 | Spearman, *Beyond Expected Goals* / *Physics-Based Modeling of Pass Probabilities* (MIT SSAC) | Pitch control model |
| 16 | StatsBomb Open Data — github.com/statsbomb/open-data | xG fit + calibration + Option B |
| 17 | Friends of Tracking, *LaurieOnTracking* — github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking | Reference pitch-control / EPV implementation |
| 18 | Metrica Sports sample data — github.com/metrica-sports/sample-data | Open tracking data for validation |
| 19 | Raffin et al., *Stable-Baselines3*, JMLR 22 (2021); `sb3-contrib` MaskablePPO | Implementation |
| 20 | Towers et al., *Gymnasium* (Farama Foundation) | Environment API |

---

## 11. Immediate next steps

1. **Commit to GAFFER now, with Gate 1 as the safety net.** Phase 0 is option-agnostic, so this commitment costs nothing if you later fall back — which is exactly why it can be made immediately rather than deliberated.
2. Take this document to the consultation and get the stakeholder sign-off on the custom-environment route — the brief explicitly permits it ("developing a custom OpenAI Gym environment... and use RL methods to solve it").
3. Split ownership (§6.1) in writing, in the repo README, on day one, so each layer has a clear owner.
4. **A: write the stub PITCHSIM before writing any physics.** Freeze the observation vector and the replay JSON schema, version them, and hand them to B and C. This is the highest-leverage hour in the whole project.
5. Start the StatsBomb loader + xG model. Self-contained, useful under *both* options, and it de-risks the calibration story before a single line of physics exists.
6. Write deliverable from §3's elevator pitch.
7. Decide the JAX question (§6.4) now or never — it is a Phase-0 commitment, and the default answer is no.

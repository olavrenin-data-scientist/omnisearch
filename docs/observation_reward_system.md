# Observation and Reward System

This document explains the observation and reward design used by the current
OmniSearch HAPPO default. It treats the following saved training run as the
reference configuration:

```text
results/harl_runs/wildfire/wildfire_search/happo/
happo_uav4_ugv3_area_1sqkm_malibu_grid256_steps900_fire_survivors_10/
seed-00001-2026-07-19-00-15-35/omnisearch_training_config.json
```

The reference run uses 4 UAVs, 3 UGVs, 10 survivors, a 1 km x 1 km Malibu Creek
terrain, a 256 x 256 fire/perception grid, active fire dynamics, and 900-step
episodes.

## Design Goal

The observation and reward system is built around the operational structure of
wildfire search and rescue:

1. UAVs reduce uncertainty over the search area.
2. Scouted survivor locations become actionable targets.
3. UGVs use terrain- and fire-aware navigation to confirm those targets.

The policy should therefore learn more than geometric coverage. It should learn
when an aerial revisit is useful, when it is redundant, when a ground robot
should move, and how the two roles create a completed mission together.

## Observation: What Each Agent Knows

Each HAPPO actor receives its own local observation. The centralized critic
receives the concatenation of all agent observations during training, as
described in [docs/reinforcement_learning_happo.md](reinforcement_learning_happo.md).

The observation is not intended to be a raw simulator state dump. It is a compact
task representation with four conceptual layers:

- **Physical state:** where the agent is, how it is moving, and where the map
  boundaries are.
- **Local traversability and hazard context:** what terrain, clearance, and fire
  conditions surround the agent.
- **Team and survivor information:** where teammates are and which survivors are
  currently known or confirmed.
- **Search memory:** where the aerial team still has low detection confidence.

In the reference run, the flat per-agent observation has width 1351. Most of
this width is the global confidence map, because the UAV search problem is
represented as probabilistic inspection memory rather than a single binary
covered/uncovered layer.

| Conceptual layer | Main fields | Default size |
|---|---|---:|
| Physical state | position, velocity, boundary distances, flight state | 10 |
| Local sensing and terrain | lidar/range placeholder, fire density, mobility patch, air-clearance patch | 120 |
| Team context | teammate relative positions | 12 |
| Survivor task state | 10 survivor messages with assignment flags | 90 |
| Search memory | 32 x 32 global confidence, 9 x 9 local confidence, local/global frontier | 1114 |
| UGV route guidance | global A* waypoint hint | 5 |

The total is:

```text
10 + 120 + 12 + 90 + 1114 + 5 = 1351
```

### Survivor Messages

Survivors are hidden until discovered. The observation therefore does not expose
all survivor positions from reset. Instead, each of the 10 survivor slots is
inactive until the survivor is known to the receiving agent.

Each active survivor message contains relative geometry and task status:

```text
[known, dx, dy, ux, uy, distance_norm, confirmed,
 assigned_to_me, assigned_to_other_ugv]
```

The first seven values say whether the survivor is known, where it is relative
to the agent, how far away it is, and whether it has already been confirmed. The
last two values expose UGV assignment state. This lets a UGV focus on its own
target and lets UAVs see whether a discovered survivor has already been handed
off to the ground team.

### Confidence Memory

The main search-memory variable is the confidence map $C_t(x)$. It is the
cumulative probability that a survivor at cell $x$ would already have been
detected by the UAV team.

Low confidence means "this cell still deserves inspection." High confidence
means "if a survivor were here, we likely would have detected them already."

The reference run gives the policy two views of this memory:

- A **global 32 x 32 confidence map** plus global mean confidence.
- A **local 9 x 9 confidence patch** around the agent, covering a 60 m radius.

This lets UAVs reason at two scales. The global map supports broad allocation
across the full terrain. The local patch supports fine decisions about whether a
nearby pass is still valuable.

### Confidence Frontier

The frontier observation converts confidence memory into a directional search
hint. It does not replace the confidence map; it summarizes where nearby and
global low-confidence opportunity lies.

The reference run uses a local/global frontier with 8 features:

```text
[local_dx, local_dy, local_distance, local_score,
 global_dx, global_dy, global_distance, global_score]
```

The local candidate looks within 60 m. The global candidate searches over the
larger map and uses ownership weighting so multiple UAVs do not all choose the
same remaining low-confidence region when alternatives exist.

### UGV Planner Hint

UGVs receive a compact global A* route hint toward their assigned target. The
planner uses terrain, land cover, and fire-aware traversability. In the reference
run, sufficiently active fire is treated as blocked for route construction, and
routes are lazily replanned as the fire evolves.

Conceptually, the planner hint tells the learned UGV policy: "the direct line to
the survivor may be physically wrong; here is the next feasible direction."

### Role-Specific Critic Masks

The reference HAPPO setup shares actor parameters by role: one UAV actor for all
UAVs and one UGV actor for all UGVs. The critic can also use role-specific masks.
The UAV critic ignores the UGV planner hint, while the UGV critic ignores
UAV-only search-map fields. This keeps the training signal role-relevant without
changing the environment's flat observation interface.

## Reward: What Behavior Is Reinforced

The reward system is a credit-assignment mechanism for the UAV-to-UGV workflow.
It combines sparse mission events with dense role-specific signals:

```text
agent reward
  = shared mission progress
  + role-specific event credit
  + dense guidance
  - wasted effort / hazard penalties
```

The reference configuration deliberately disables binary coverage reward. UAVs
are instead trained on probabilistic confidence gain, because a cell is not just
"covered" or "uncovered": detection quality depends on altitude, footprint
position, terrain, smoke, and fire.

## Shared Mission Progress

The team receives reward when a survivor is newly scouted and when a survivor is
newly confirmed. With the reference run's `comms_dropout = 0.0`, the shared team
event reward reaches every agent.

| Event | Weight | Interpretation |
|---|---:|---|
| Newly scouted survivor | 1.0 | the aerial team made a survivor actionable |
| Newly confirmed survivor | 4.0 | the team completed a survivor-confirmation objective |

The UGV that physically confirms the survivor also receives an individual
confirmation reward of 10.0. This separates team value from local role credit:
everyone benefits when the mission advances, but the confirming UGV receives
strong direct credit for completing the ground task.

## UAV Reward Concept

The UAV reward answers a specific question:

> Did this UAV use its camera footprint to reduce meaningful search uncertainty?

The confidence map is updated from per-cell detection probabilities
$p_{i,t}(x)$:

$$
C_{t+1}(x)
=
1 -
\left(1-C_t(x)\right)
\prod_i \left(1-p_{i,t}(x)\right)
$$

Equivalently, the remaining miss probability is:

$$
1-C_{t+1}(x)
=
\left(1-C_t(x)\right)
\prod_i \left(1-p_{i,t}(x)\right).
$$

Each UAV receives marginal credit for how much its footprint improves this
confidence map. The gain is weighted toward uncertain cells:

$$
w_t(x) = \epsilon + \left(1-C_t(x)\right)^\gamma
$$

with $\epsilon=0.05$, $\gamma=2.0$, and confidence reward weight 30.0. This
makes low-confidence cells much more valuable than repeatedly inspecting areas
that are already likely clear.

The UAV also receives small guidance and efficiency terms:

| Term | Weight | Conceptual role |
|---|---:|---|
| Direct scout credit | 2.0 | credit the UAV that newly detects a survivor |
| Confidence gain | 30.0 | reward marginal improvement to probabilistic inspection confidence |
| Confidence-gated movement | 0.1 | reward motion only when it captures confidence opportunity |
| Frontier alignment | 0.05 | guide UAVs toward local/global low-confidence regions |
| Inefficient movement | -0.005 scale | mild energy-style penalty for moving without useful confidence gain |

The movement terms are an energy proxy. They do not pay UAVs merely for moving.
Instead, movement is useful only when it creates new inspection value, and
wasteful motion over low-opportunity regions is mildly discouraged.

## UAV Penalty Concept

The UAV penalties discourage ways of accumulating poor or misleading search
coverage:

- **Confidence overlap:** revisiting high-confidence cells is penalized when
  better confidence opportunity was available.
- **Outside-footprint loss:** camera area outside the map is wasted field of
  view and receives a penalty.
- **Fire-footprint loss:** camera area over active fire receives a penalty,
  because burning regions are poor survivor-search targets and can distort
  perception.

The active fire-related UAV term is:

$$
r^{\mathrm{fire\_footprint}}_{i,t}
=
-0.05 \cdot f^{\mathrm{fire}}_{i,t},
$$

where $f^{\mathrm{fire}}_{i,t}$ is the fraction of the UAV footprint covering
fire cells with intensity at least 0.6.

In the reference configuration, binary overlap and inter-UAV overlap penalties
are disabled. The confidence-overlap penalty is preferred because overlap is not
always bad: revisiting a low-confidence smoky or edge cell can still be useful.

## UGV Reward Concept

The UGV reward answers a different question:

> Once a survivor is known, did this UGV make feasible progress toward
> confirmation?

UGVs cannot act meaningfully on hidden survivors. Once a survivor is scouted,
the reward encourages a UGV to move along a feasible route and confirm it within
the 10 m confirmation radius.

The reference run uses `planner_follow` mode. Progress is measured against the
global A* route waypoint rather than a straight line to the survivor. This is
important in wildfire terrain: a straight path may cross blocked terrain, steep
terrain, water, or active fire. The reward therefore teaches the policy to
follow a route that respects the simulated physical constraints.

The active UGV terms are:

| Term | Weight | Conceptual role |
|---|---:|---|
| Ground confirmation | 10.0 | credit the UGV that confirms a survivor |
| Planner-follow progress | 0.5 | reward progress along the feasible A* route |
| Movement alignment | 0.2 | reward displacement aligned with the planner direction |
| Pending target pressure | -0.02 | discourage leaving known assigned targets unresolved |
| Route shortfall penalty | -0.0025 scale | penalize falling behind the progress needed to finish in time |

The pending penalty is assignment-aware. A standby UGV is not punished for a
target already assigned to another UGV. This matters because the ground team can
have fewer useful targets than robots at some points in the episode.

## Fire Penalty and Fire-Aware Routing

There are two fire-related reward mechanisms:

1. UAVs receive the active fire-footprint penalty described above.
2. UGVs can receive a direct fire-exposure penalty if `r_fire_penalty` is set
   below zero.

The saved reference config has `r_fire_penalty = 0.0`, so the direct UGV
fire-exposure reward term is currently disabled:

$$
r^{\mathrm{ground\_fire}}_{g,t}
=
r_{\mathrm{fire\_penalty}}
\cdot \mathbf{1}\{\text{UGV } g \text{ is in fire}\}.
$$

Even with that direct penalty disabled, fire still affects UGV behavior through
planning. The default route planner uses `ugv_planner_fire_mode = block`, so
active fire can make route cells unavailable. In other words, fire is already
part of the UGV navigation problem; `r_fire_penalty` would add an extra learned
cost for physically being in fire.

## Default Values Used Here

The most important active weights in the reference configuration are:

| Group | Active weights |
|---|---|
| Team events | scout 1.0, confirmation 4.0 |
| UAV events/search | scout 2.0, confidence 30.0, confidence movement 0.1, frontier 0.05 |
| UAV penalties | confidence overlap 0.06, outside footprint 0.1, fire footprint 0.05, inefficient movement 0.005 |
| UGV progress | ground confirmation 10.0, planner-follow shaping 0.5, movement alignment 0.2 |
| UGV pressure | pending target -0.02, route shortfall 0.0025 |

The main disabled terms are binary coverage reward, binary overlap penalties,
team confidence reward, cleanup-target progress reward, UAV A* progress reward,
ground travel cost, stall penalty, and direct UGV fire exposure.

## Practical Interpretation

The reward system encodes a division of labor:

- UAVs should search where detection confidence is still low, not merely cover
  the largest possible area.
- UAVs should avoid high-confidence revisits, map-edge waste, and active-fire
  footprints unless those regions still offer meaningful uncertainty reduction.
- UGVs should not wait passively once a survivor is known; they should follow a
  feasible fire-aware route and confirm the target.
- Team rewards couple both roles, so the learned behavior is measured by mission
  completion rather than by isolated aerial or ground motion.

The intended learned strategy is therefore: probabilistic aerial discovery,
assignment-aware handoff, route-aware ground confirmation, and efficient use of
the limited episode horizon.

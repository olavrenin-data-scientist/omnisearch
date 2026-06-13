"""
Builds docs/code_walkthrough.pdf — a per-file, code-block-by-code-block
explanation of every source file in the project (Python + the web viewer +
the notebooks).

Approach: each source file gets a section with the file's purpose, then we
walk the file in code blocks (functions, classes, key statements) and write
prose for each. The walkthrough is authored as a lightweight Markdown string;
a small renderer turns it into ReportLab flowables (headings, paragraphs,
syntax-highlighted code blocks via Pygments) and writes the PDF.

ReportLab is used instead of WeasyPrint because it is pure-Python and needs
no native libraries (Pango/Cairo/GObject), so the build works on any machine
that can `pip install reportlab` — no Homebrew system packages required.

Run:
    .venv/bin/python docs/build_walkthrough.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from pygments import lex
from pygments.lexers import (
    HtmlLexer,
    JavascriptLexer,
    PythonLexer,
    TextLexer,
)
from pygments.styles import get_style_by_name

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    XPreformatted,
)

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Content — the walkthrough text itself. Markdown with fenced code blocks.
# ----------------------------------------------------------------------
WALKTHROUGH = r"""
# OmniSearch — Code Walkthrough

**Heterogeneous air–ground swarm for wildfire survivor search.**
MIDS Capstone · UC Berkeley.

This document walks every Python source file in the project. Each file
gets:

1. **Purpose** — one paragraph on what the file is for and how it fits the
   rest of the system.
2. **Code, block by block** — the actual source with prose between blocks
   explaining what each piece is doing and *why* the choice was made.

Files are grouped by package. Trivial code (imports, boilerplate
docstrings, single-line CLI argparse plumbing) is summarised; non-obvious
logic gets a paragraph or two.

## Architecture at a glance

```
                          MARL  (the research field)
                            │
              ┌─────────────┼─────────────┐
              │             │             │
            MAPPO         IPPO          HAPPO          ← three algorithms,
         (BenchMARL)   (BenchMARL)     (HARL)            same scenario
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                ┌─────────────────────┐
                │    VMAS simulator   │  ← the heart of everything
                │  WildfireSearch...  │
                └─────────────────────┘
```

The scenario is implemented once in `envs/wildfire_search.py`. Every
training algorithm (MAPPO, IPPO via BenchMARL; HAPPO via HARL) and every
evaluation tool (mission metrics, baseline comparison, trajectory export,
web viewer) consumes it through different adapters. The rest of the code
is glue: turning VMAS's tensor-batched API into the shape each downstream
library expects.

\\newpage

# Package: `envs`

The scenario class. Subclasses VMAS's `BaseScenario` so VMAS treats it
like any other scenario (with `make_world`, `reset_world_at`,
`observation`, `reward`, `done`).

## `envs/wildfire_search.py`

Defines `WildfireSearchScenario` — heterogeneous drone + ground robot
search for survivors in a spreading wildfire. Drones scout fast and wide;
ground robots verify precisely and pay a penalty for entering burning
cells.

### Header and imports

```python
'''OmniSearch Custom VMAS Scenario: Wildfire Survivor Search…'''

from __future__ import annotations
from typing import Callable, Dict, List
import torch
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

X, Y = 0, 1
```

Standard VMAS imports. `Agent`, `Landmark`, `World` are the building
blocks; `BaseScenario` is the interface we implement; `Lidar` is the
sensor we attach to agents to give them visibility of survivors;
`ScenarioUtils.spawn_entities_randomly` handles non-overlapping random
spawn. `X = 0, Y = 1` are just index aliases so position-vector indexing
reads naturally (`pos[X]`, `pos[Y]`).

### `class WildfireSearchScenario(BaseScenario)`

A VMAS scenario must define `make_world`, `reset_world_at`, `observation`,
`reward`, `done`. Optionally `pre_step` (called once per env-step before
physics) and `info`. We use all of them.

### `make_world(batch_dim, device, **kwargs)`

Called once when the env is created. Returns a `World` populated with
agents and landmarks; also stashes scenario-level state tensors. Every
parameter accepted by `make_world` is a *scenario kwarg* that callers
(BenchMARL task config, HARL adapter, comms-dropout sweep) can override.

```python
self.n_drones    = kwargs.pop("n_drones", 3)
self.n_ground    = kwargs.pop("n_ground", 2)
self.n_survivors = kwargs.pop("n_survivors", 5)
self.n_agents    = self.n_drones + self.n_ground
```

Team composition per project plan: 3 aerial scouts + 2 ground verifiers
searching for 5 survivors. `n_agents` is a convenience.

```python
self.x_semidim = kwargs.pop("x_semidim", 1.0)
self.y_semidim = kwargs.pop("y_semidim", 1.0)
```

World extends `[-1, 1]` in both axes. VMAS uses this for boundary
enforcement and rendering.

```python
self.drone_lidar_range  = kwargs.pop("drone_lidar_range",  0.50)
self.ground_lidar_range = kwargs.pop("ground_lidar_range", 0.20)
self.n_lidar_rays       = kwargs.pop("n_lidar_rays", 12)
self.detection_range    = kwargs.pop("detection_range", 0.13)
```

Drones see survivors out to 0.5 (half the world); ground robots only 0.2
(narrower, more precise sensors). Ground robots confirm a survivor when
within `detection_range=0.13` — slightly bigger than the original 0.10 so
that approach + confirm doesn't require pin-point alignment.

```python
self.fire_grid_size      = kwargs.pop("fire_grid_size", 16)
self.fire_spread_prob    = kwargs.pop("fire_spread_prob", 0.04)
self.initial_fire_cells  = kwargs.pop("initial_fire_cells", 1)
self.fire_step_interval  = kwargs.pop("fire_step_interval", 5)
```

The fire grid is a discrete 16×16 overlay over the continuous world. Each
burning cell can spread to a neighbor with probability `fire_spread_prob`.
Spread is gated to every `fire_step_interval` env-steps so the fire grows
at a controlled pace.

```python
self.comms_dropout = kwargs.pop("comms_dropout", 0.0)
self.max_steps = kwargs.pop("max_steps", 400)
```

`comms_dropout` is the per-step probability that an agent's
teammate-observation slot gets zeroed — the radio failure model studied
in the comms ablation. `max_steps` is the episode horizon (400 since the
relaxed scenario commit).

```python
self.r_found_survivor = kwargs.pop("r_found_survivor", 1.0)
self.r_drone_scout    = kwargs.pop("r_drone_scout", 0.3)
self.r_ground_confirm = kwargs.pop("r_ground_confirm", 0.5)
self.r_time_penalty   = kwargs.pop("r_time_penalty", -0.001)
self.r_fire_penalty   = kwargs.pop("r_fire_penalty", -1.0)
self.r_drone_shaping  = kwargs.pop("r_drone_shaping",  0.05)
self.r_ground_shaping = kwargs.pop("r_ground_shaping", 0.10)
```

The reward weights. Sparse rewards: `r_found_survivor` for the team when
a new survivor is confirmed; `r_drone_scout` to the drone that first
scouts each survivor; `r_ground_confirm` to the ground robot that
confirms one. Penalties: tiny per-step time penalty to pressure the
agents to hurry; large fire penalty applied each step a ground robot
stands on a burning cell. The two `*_shaping` weights are the dense
potential-based shaping added later — α·(prev_dist − curr_dist) per
agent — that gives a gradient signal toward the nearest live target.

```python
ScenarioUtils.check_kwargs_consumed(kwargs)
```

VMAS helper: if the caller passed any kwarg we didn't `.pop()`, raise
immediately. Catches typos like `n_drone=3` vs `n_drones=3`.

```python
world = World(batch_dim, device, x_semidim=self.x_semidim,
              y_semidim=self.y_semidim, collision_force=300,
              substeps=2, drag=0.25)
```

Build the VMAS world. `batch_dim` is the number of parallel environments
(VMAS batches everything). `collision_force` and `drag` are standard
physics knobs; `substeps=2` does 2 physics substeps per env-step for
stability.

```python
survivor_filter: Callable = lambda e: e.name.startswith("survivor")
```

Lidar uses an `entity_filter` to decide which entities it can hit. By
filtering on the name prefix, drones'/ground's lidar only sees survivors —
not other agents, not fire (fire isn't an entity, it's a tensor overlay).

#### Drone construction loop

```python
for i in range(self.n_drones):
    agent = Agent(
        name=f"drone_{i}",
        collide=True,
        shape=Sphere(radius=self.agent_radius),
        max_speed=0.5,
        u_range=1.0, u_multiplier=0.6,
        color=Color.BLUE,
        sensors=[Lidar(world, n_rays=self.n_lidar_rays,
                       max_range=self.drone_lidar_range,
                       entity_filter=survivor_filter,
                       render_color=Color.RED)],
    )
    agent.is_drone = True
    world.add_agent(agent)
```

Each drone is a VMAS `Agent` with `max_speed=0.5` and a single lidar
sensor. `u_range=1.0, u_multiplier=0.6` defines the action-to-force
mapping: actions are clipped to `[-1, 1]` then scaled by `0.6` to compute
the applied force. `agent.is_drone = True` is a custom attribute we set
so reward/observation code can distinguish drone vs ground without
matching on the name.

#### Ground robot construction loop

```python
for i in range(self.n_ground):
    agent = Agent(
        name=f"ground_{i}",
        collide=True, shape=Sphere(radius=self.agent_radius),
        max_speed=0.2, u_range=1.0, u_multiplier=0.3,
        color=Color.GREEN, sensors=[…])
    agent.is_drone = False
    world.add_agent(agent)
```

Same pattern but `max_speed=0.2` (2.5× slower than drones) and narrower
lidar (`ground_lidar_range=0.20`). The ground robots are the bottleneck of
the system — they need to physically reach scouted survivors to confirm.

#### Survivors

```python
self._survivors: List[Landmark] = []
for i in range(self.n_survivors):
    survivor = Landmark(name=f"survivor_{i}",
                        collide=True, movable=False,
                        shape=Sphere(radius=self.survivor_radius),
                        color=Color.RED)
    world.add_landmark(survivor)
    self._survivors.append(survivor)
```

Survivors are stationary `Landmark`s. `movable=False` means physics
doesn't push them around when an agent collides. Stored both in the
world (so they render) and in `self._survivors` (so reward/observation
code can iterate them).

#### Scenario state tensors

```python
self.found_survivors = torch.zeros(
    batch_dim, self.n_survivors, dtype=torch.bool, device=device)
self.scouted_survivors = torch.zeros_like(self.found_survivors)
self.fire_grid = torch.zeros(batch_dim, self.fire_grid_size,
    self.fire_grid_size, dtype=torch.bool, device=device)
self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
```

Per-batch state. `scouted_survivors[b, i] = True` once any drone has
lidar'd survivor `i` in env `b`. `found_survivors[b, i] = True` once a
ground robot has reached confirm radius. `fire_grid[b, gy, gx]` is the
discrete burning-cell mask. `step_count` tracks the per-env timestep.

```python
self.prev_drone_dist = torch.full(
    (batch_dim, self.n_drones), float("inf"), device=device)
self.prev_ground_dist = torch.full(
    (batch_dim, self.n_ground), float("inf"), device=device)
```

Previous-step minimum distance from each agent to its target type. Used
by the dense shaping reward — see `_compute_step_rewards` below.
Initialised to `+inf` so the first step yields zero shaping (we don't
have a reference yet).

```python
for agent in world.agents:
    agent.scenario_reward = torch.zeros(batch_dim, device=device)
return world
```

Per-agent reward buffer, allocated once. `reward()` fills it on each call
to the first agent's reward (see below).

### `reset_world_at(env_index=None)`

Called by VMAS on env reset. Spawns entities at random non-overlapping
positions and clears per-batch state.

```python
ScenarioUtils.spawn_entities_randomly(
    entities=self._survivors + self.world.agents,
    world=self.world, env_index=env_index,
    min_dist_between_entities=2*self.agent_radius + 0.02,
    x_bounds=(-self.x_semidim, self.x_semidim),
    y_bounds=(-self.y_semidim, self.y_semidim))
```

`env_index=None` means reset every env in the batch; passing an index
resets just that one (used by the batched HARL adapter for auto-reset on
episode termination).

The rest of `reset_world_at` zeros the state tensors and seeds the initial
fire cells (`initial_fire_cells` random cells set to True).

### `pre_step(self)`

Called by VMAS before each env-step, before agent actions are applied.
Our hook spreads the fire on a cellular-automata schedule:

```python
self.step_count += 1
if int(self.step_count.max().item()) % self.fire_step_interval != 0:
    return
fire = self.fire_grid.float()
padded = torch.zeros(fire.shape[0], fire.shape[1]+2, fire.shape[2]+2, …)
padded[:, 1:-1, 1:-1] = fire
neighbors = (padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1]
           + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:])
p_ignite = 1.0 - (1.0 - self.fire_spread_prob) ** neighbors
new_burns = torch.rand_like(p_ignite) < p_ignite
self.fire_grid = self.fire_grid | new_burns
```

Counts the 4-neighbor burning cells via shifted slices of a padded copy.
The ignition probability for an unburnt cell with k burning neighbors is
`1 − (1 − p)^k` — the standard "k independent chances to ignite" formula.

### `reward(agent)`

Called once per agent per env-step. The first call per step computes
*all* per-agent rewards in one batched pass; later calls just return the
cached value.

```python
def reward(self, agent):
    if agent is self.world.agents[0]:
        self._compute_step_rewards()
    return agent.scenario_reward
```

### `_compute_step_rewards(self)` — the hot path

The function that produces the reward signal HAPPO/MAPPO/IPPO trains on.

```python
agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
surv_pos  = torch.stack([s.state.pos for s in self._survivors], dim=1)
dists = torch.cdist(agent_pos, surv_pos)
```

Batched pairwise distances: `dists[b, a, s]` is the distance from agent
`a` to survivor `s` in env `b`. `torch.cdist` is the canonical pairwise
Euclidean distance op.

```python
lidar_ranges = torch.tensor([self.drone_lidar_range] * self.n_drones
                          + [self.ground_lidar_range] * self.n_ground,
                          device=device).view(1, self.n_agents, 1)
seen = dists < lidar_ranges
```

Per-agent visibility mask. Each row of `lidar_ranges` is broadcast to
match `dists`, so `seen[b, a, s]` is True iff agent `a` is within its own
lidar range of survivor `s`.

```python
seen_by_drone       = seen[:, :self.n_drones, :].any(dim=1)
within_confirm      = dists < self.detection_range
confirmed_by_ground = within_confirm[:, self.n_drones:, :].any(dim=1)
```

Aggregate per survivor: `seen_by_drone[b, s]` is True iff *any* drone
sees survivor `s` in env `b`. `confirmed_by_ground[b, s]` is True iff
*any* ground robot is within confirm radius.

```python
newly_scouted = seen_by_drone & ~self.scouted_survivors & ~self.found_survivors
newly_found   = confirmed_by_ground & ~self.found_survivors
self.scouted_survivors |= newly_scouted
self.found_survivors   |= newly_found
```

A scout/confirm event fires only on the *first* step it becomes true —
that's why we mask against `~self.scouted_survivors` etc. After computing
the events, we OR them into the persistent state.

```python
team_reward = (newly_found.float().sum(dim=1) * self.r_found_survivor
             + self.r_time_penalty)
```

Shared reward across the team: `+1` per newly confirmed survivor, with
the small per-step time penalty.

#### Per-agent credit

```python
drone_seen        = seen[:, :self.n_drones, :]
scout_credit_mask = drone_seen & newly_scouted.unsqueeze(1)
scout_per_drone   = scout_credit_mask.float().sum(dim=2)
```

Which drone gets the scout bonus? Anyone who personally saw the
newly-scouted survivor. If multiple drones see the same survivor on the
first step it's scouted, they each get a unit of credit (sums to ≥ 1 per
fresh survivor).

```python
ground_within        = within_confirm[:, self.n_drones:, :]
confirm_credit_mask  = ground_within & newly_found.unsqueeze(1)
confirm_per_ground   = confirm_credit_mask.float().sum(dim=2)
ground_in_fire = self._agents_in_fire(self.world.agents[self.n_drones:])
```

Same idea for ground robots. `ground_in_fire` is a separate helper that
returns a (B, G) boolean of whether each ground robot is on a burning
cell this step — used for the per-step fire penalty.

#### Dense shaping rewards

```python
unscouted = ~self.scouted_survivors
drone_d = torch.where(unscouted.unsqueeze(1),
                     dists[:, :self.n_drones, :], torch.full_like(…, INF))
curr_drone_dist, _ = drone_d.min(dim=2)
all_scouted = torch.isinf(curr_drone_dist)
curr_drone_dist = torch.where(all_scouted,
                              torch.zeros_like(curr_drone_dist),
                              curr_drone_dist)
prev_known = ~torch.isinf(self.prev_drone_dist) & ~all_scouted
drone_shaping = torch.where(prev_known,
    (self.prev_drone_dist - curr_drone_dist) * self.r_drone_shaping,
    torch.zeros_like(curr_drone_dist))
self.prev_drone_dist = curr_drone_dist
```

The drone-side shaping. We mask `dists` so non-target distances become
`+inf`, then `min` gives the distance to the nearest *unscouted*
survivor. `prev_known` guards against two bad cases: (1) the first step
where there's no `prev_dist` yet, (2) all survivors are scouted and there
are no live targets. Shaping is `(prev - curr) * α` — positive when the
agent moved closer to its target. We then update `prev_dist = curr_dist`
for the next step.

The ground-side shaping is identical but targets `scouted &
~confirmed`, with weight `r_ground_shaping = 0.10`.

This is the Ng et al. (1999) potential-based form which preserves the
optimal policy of the original reward — useful theoretical property since
it means the shaping doesn't change *what* a perfect policy looks like,
only how fast a learner converges to it.

#### Per-agent reward composition

```python
for i, agent in enumerate(self.world.agents):
    r = team_reward.clone()
    if agent.is_drone:
        r = r + scout_per_drone[:, i] * self.r_drone_scout
        r = r + drone_shaping[:, i]
    else:
        g = i - self.n_drones
        r = r + confirm_per_ground[:, g] * self.r_ground_confirm
        r = r + ground_in_fire[:, g].float() * self.r_fire_penalty
        r = r + ground_shaping[:, g]
    agent.scenario_reward = r
```

Each agent gets the team reward plus its own role-specific bonuses. We
write the result into the pre-allocated `agent.scenario_reward` buffer
so `reward(agent)` later in the step just returns it without
recomputation.

### `observation(agent)`

Returns the agent's per-step observation vector — what the policy network
sees.

```python
own_pos    = agent.state.pos
own_vel    = agent.state.vel
lidar_obs  = agent.sensors[0].measure()
fire_local = self._local_fire_density(agent)
neighbor   = self._neighbor_observations(agent)
return torch.cat([own_pos, own_vel, lidar_obs, fire_local, neighbor], dim=-1)
```

A 25-dim vector per agent: position (2) + velocity (2) + lidar (12) +
local fire density (1) + relative positions of every other agent (8 = 4
neighbors × 2 coords). `lidar_obs` comes for free from the `Lidar` sensor
attached at construction.

### `_local_fire_density(agent)`

```python
gx = ((pos[..., X] + self.x_semidim) / (2*self.x_semidim) * self.fire_grid_size)…
gy = ((pos[..., Y] + self.y_semidim) / (2*self.y_semidim) * self.fire_grid_size)…
density = sum over (dy, dx) in 3×3 of self.fire_grid[b_idx, gy+dy, gx+dx]
return (density / 9.0).unsqueeze(-1)
```

Maps the agent's continuous position to a grid index, then averages the
3×3 window of fire cells around it. Gives the agent a scalar "how much
fire near me right now" feature — important for ground robots that need
to avoid burning cells.

### `_neighbor_observations(agent)`

The communication channel — what each agent knows about its teammates.

```python
deltas = []
for other in self.world.agents:
    if other is agent: continue
    deltas.append(other.state.pos - agent.state.pos)
rel = torch.cat(deltas, dim=-1)
if self.comms_dropout > 0:
    keep_bool = torch.rand_like(rel[..., :1]) > self.comms_dropout
    rel = rel * keep_bool.float()
    agent.comms_up = keep_bool.squeeze(-1)
else:
    agent.comms_up = torch.ones(self.world.batch_dim, dtype=torch.bool, …)
return rel
```

The communication model: each agent observes the *relative position* of
every other agent. With `comms_dropout > 0`, the entire teammate slot of
that agent's obs zero's out with that probability — the "radio dropped
this step" event. We record `agent.comms_up` (a bool per env) so the
trajectory exporter and viewer can draw a green/red dot above each agent.

### `done(self)`

```python
return self.found_survivors.all(dim=1) | (self.step_count >= self.max_steps)
```

Episode ends when all survivors are confirmed *or* the step budget is
exhausted.

### `info(self, agent)`

Optional VMAS hook that returns a per-agent dict of debug values. We
expose three scenario-level counters that downstream metrics use:

```python
return {"n_found": …, "n_scouted": …, "n_burning": …}
```

\\newpage

# Package: `agents`

The code that wires the scenario to MARL libraries and to evaluation.

## `agents/wildfire_task.py`

BenchMARL integration. Subclasses `VmasClass` so BenchMARL's stock VMAS
training path — `Experiment(task=…)` — works on our custom scenario.

```python
class WildfireVmasClass(VmasClass):
    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        config = copy.deepcopy(self.config)
        return lambda: VmasEnv(
            scenario=WildfireSearchScenario(),
            num_envs=num_envs, continuous_actions=continuous_actions,
            seed=seed, device=device,
            categorical_actions=True, clamp_actions=True, **config)
```

The key override: BenchMARL's stock `VmasClass.get_env_fun` instantiates
the scenario by *name* (string lookup against VMAS's built-in registry).
We override it to pass a scenario *instance* instead. Everything else —
the `Task` enum, the `algorithm_config`, the `model_config` — works
unchanged because BenchMARL is structured around polymorphism here.

```python
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_steps": 400, "n_drones": 3, "n_ground": 2, "n_survivors": 5,
    "x_semidim": 1.0, "y_semidim": 1.0,
    "drone_lidar_range": 0.50, "ground_lidar_range": 0.20,
    "n_lidar_rays": 12, "detection_range": 0.13, …
}
```

Default scenario config matching the relaxed-recall version. Callers can
override with `make_wildfire_task(**overrides)`.

```python
def make_wildfire_task(**overrides):
    config = {**DEFAULT_CONFIG, **overrides}
    return WildfireVmasClass(name="WILDFIRE_SEARCH", config=config)
```

The entry point used by all BenchMARL-based training scripts.

## `agents/train_helpers.py`

Shared `ExperimentConfig` builders so MAPPO and IPPO smoke scripts don't
duplicate the same knob settings. Two factory functions:

```python
def smoke_config(iters=3, frames_per_batch=2_000, envs_per_worker=8, …):
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = iters
    cfg.on_policy_collected_frames_per_batch = frames_per_batch
    cfg.on_policy_n_envs_per_worker = envs_per_worker
    cfg.lr = 3e-4
    cfg.evaluation = False
    cfg.render = False
    cfg.save_folder = str(ROOT / "results")
    return cfg

def research_config(iters=200, frames_per_batch=6_000, envs_per_worker=32, …):
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = iters
    cfg.evaluation = True
    cfg.checkpoint_at_end = True
    …
    return cfg
```

Smoke is for "does it train at all?" verification; research is for the
actual numbers you write up. The split is intentional — researchers
shouldn't have to tweak a dozen knobs to switch budgets.

## `agents/baselines.py`

Four hand-coded coordination strategies that the trained policies must
beat to be defensible. Each implements the same callable signature
`policy(env) → list of per-agent action tensors`, so they all plug into
the same evaluation harness.

```python
class RandomActionPolicy:
    def __call__(self, env):
        return env.get_random_actions()
```

The control. `env.get_random_actions()` samples uniform-in-range actions
for every agent. Establishes the lower bound — anything that beats
random has *some* coordination value.

```python
class LawnmowerPolicy:
    def __init__(self, env):
        self.scenario = env.scenario
        self.t = 0
        self.drone_band_y = self._drone_band_ys(self.scenario.n_drones)

    def __call__(self, env):
        # drones: serpentine sweep, each drone owns a y-band
        # ground: head to nearest scouted survivor
```

The lawnmower: drones do horizontal sweeps at different `y` values; each
drone owns a band and reverses direction every `period` steps. Ground
robots head to the nearest `scouted & ~confirmed` survivor. The drone
trajectories are deterministic but the survivor placement is random, so
recall varies seed-to-seed.

```python
class NearestCandidatePolicy:
    def __call__(self, env):
        # drones: random
        # ground: nearest scouted survivor
```

Like lawnmower but drones don't do area coverage — they just wander
randomly. This isolates *what fraction of recall comes from smart ground
behaviour alone?* Empirically, about the same as lawnmower — most of the
win is the ground policy.

```python
class HighestConfidencePolicy:
    def __init__(self, env):
        self.scout_step = [-1] * self.scenario.n_survivors
        self.t = 0
        self._lawnmower = LawnmowerPolicy(env)

    def __call__(self, env):
        # drones: lawnmower
        # ground: target = freshest (most recently scouted) survivor
```

A proxy for the project plan's "highest-confidence candidate" policy.
Real implementation would need a per-survivor confidence score — until
the probabilistic sensor model is in, we proxy with "most recently
scouted = highest confidence" (the newest information is the most
reliable).

```python
BASELINES = {
    "random_action": RandomActionPolicy, "random_walk": RandomWalkPolicy,
    "lawnmower": LawnmowerPolicy,
    "nearest_candidate": NearestCandidatePolicy,
    "highest_confidence": HighestConfidencePolicy,
}
```

The registry. Both `scripts/compare_baselines.py` and
`scripts/export_trajectories.py` iterate this dict.

## `agents/harl_env.py`

Single-env HARL adapter. HARL expects a per-env wrapper (one Python
object = one env). This is the simple version; the batched version
`harl_vec_env.py` below is the one we actually train with.

```python
class WildfireHARLEnv:
    def __init__(self, args):
        # build VMAS env with num_envs=1
        # set HARL contract attrs: agents, n_agents, observation_space,
        # share_observation_space (concat of locals), action_space

    def step(self, actions):
        # actions: ndarray (n_agents, action_dim)
        # convert to VMAS action list of (1, action_dim) tensors
        # step VMAS, unpack obs/rew/dones/infos into HARL's expected shape

    def reset(self):
        # bump seed, rebuild env, return obs/share_obs/avail_actions
```

The translation contract is: VMAS uses batched tensors of shape `(B,
…)`; HARL uses lists/arrays without a batch dim. The adapter squeezes
the `B=1` dim out on the way in/out. `share_observation_space` is the
input to the centralised critic — we concatenate every agent's local obs
into a single global vector, then repeat once per agent.

## `agents/harl_vec_env.py`

The faster, batched HARL adapter. Wraps *one* VMAS env at `num_envs=N`
and exposes it as a `ShareVecEnv` (HARL's vec-env interface). One tensor
op per step over N envs, instead of N subprocess calls.

```python
class BatchedVMASVecEnv(ShareVecEnv):
    def __init__(self, num_envs, seed, max_cycles, scenario_kwargs, device):
        # build VMAS env with num_envs=N
        # set up obs_space/action_space/share_obs_space lists
        super().__init__(num_envs, obs_list, share_list, action_list)
        self._step_counts = np.zeros(num_envs, dtype=np.int64)

    def step_async(self, actions): self._pending_actions = actions

    def step_wait(self):
        # 1. convert actions (N, A, action_dim) → list of A tensors (N, action_dim)
        # 2. env.step → batched obs/rew/dones
        # 3. stack per-agent into ndarrays (N, A, obs_dim)
        # 4. auto-reset any done envs via scenario.reset_world_at(env_index=i)
        # 5. re-collect obs for those envs
```

Two non-obvious details:

1. **Auto-reset** — HARL expects done envs to be reset before the next
   `step_wait` returns. VMAS doesn't auto-reset, so we detect done envs
   ourselves and call `scenario.reset_world_at(env_index=i)` for each.
   Then re-collect observations for just those rows.

2. **Available actions** — HARL indexes `available_actions[0]` even in
   the continuous case to detect discrete vs continuous. Returning bare
   `None` crashes; we return an ndarray of Nones with shape `(N,)`.

The result: ~4× FPS on CPU vs the single-env adapter when running 8
parallel envs. On GPU the win is larger (probably 10–50×, untested).

## `agents/harl_runner.py`

The HAPPO training entry point. Wraps:

1. HARL env-registry monkey-patches (idempotent, safe to call many times)
2. A custom `WildfireLogger` that captures the *last* per-episode mean
   reward into an attribute — HARL's default logger clears it after each
   call, leaving callers no way to read it after `run()`.
3. `train_happo(seed, num_env_steps, comms_dropout, …)` — one call,
   returns a metrics dict.

```python
def register_wildfire_with_harl():
    '''Monkey-patches HARL's env registry to recognise env_name='wildfire'.'''
    import harl.envs as harl_envs_pkg
    import harl.utils.envs_tools as envs_tools
    import harl.utils.configs_tools as configs_tools

    def make_train_env(env_name, seed, n_threads, env_args):
        if env_name == "wildfire":
            return make_batched_wildfire_vec_env(n_threads, seed, env_args)
        return _orig_train(env_name, seed, n_threads, env_args)
    # similar for make_eval_env, make_render_env, get_num_agents,
    # configs_tools.get_task_name; install our WildfireLogger in
    # harl_envs_pkg.LOGGER_REGISTRY["wildfire"]
```

By monkey-patching at runtime rather than editing HARL source, the
project doesn't depend on a forked HARL — anyone clones PKU-MARL/HARL,
pip-installs, and our integration just works.

```python
def train_happo(seed=1, num_env_steps=8_000, comms_dropout=0.0,
                n_rollout_threads=8, exp_name="happo", entropy_coef=0.01):
    register_wildfire_with_harl()
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner

    args = {"algo": "happo", "env": "wildfire", "exp_name": exp_name, "load_config": ""}
    algo_args = default_algo_args()
    algo_args["seed"]["seed"] = seed
    algo_args["train"]["num_env_steps"]     = num_env_steps
    algo_args["train"]["n_rollout_threads"] = n_rollout_threads
    algo_args["algo"]["entropy_coef"]       = entropy_coef
    env_args = default_env_args()
    env_args["scenario_kwargs"]["comms_dropout"] = comms_dropout

    runner = OnPolicyHARunner(args, algo_args, env_args)
    runner.run()
    mean_ep = runner.logger.last_aver_episode_rewards
    runner.close()
    return {"mean_episode_reward": float(mean_ep), …}
```

Returns the metric the comms-dropout sweep aggregates.

## `agents/happo_policy.py`

Load a saved HARL checkpoint and expose it as a VMAS-style
`policy(env) → action list`. Used by the trajectory exporter to replay a
trained policy in the viewer.

```python
class HappoPolicy:
    def __init__(self, checkpoint_dir, algo_args, deterministic=True):
        # build a temp VMAS env to read obs/action spaces
        # construct one HAPPO actor per agent
        # load each actor's state_dict from checkpoint_dir/actor_agent{i}.pt
        # set actor.eval()

    def __call__(self, env):
        out = []
        for i, agent in enumerate(env.agents):
            obs = env.scenario.observation(agent).cpu().numpy()
            rnn = np.zeros((B, recurrent_n, hidden))
            masks = np.ones((B, 1))
            actions, _ = self.actors[i].act(obs, rnn, masks,
                                            available_actions=None,
                                            deterministic=True)
            out.append(torch.from_numpy(np.clip(actions, -1, 1)))
        return out
```

The non-obvious bit is that `HAPPO.act` returns `(actions, rnn_states)`
— two values, not three. The actor's neural net inside (`self.actor.actor`)
returns three (`actions, action_log_probs, rnn_states`) but the wrapper
discards the middle one. Easy thing to get wrong if you read the wrong
signature.

```python
def find_latest_happo_checkpoint(root=None):
    root = Path(root or "results/harl_runs")
    candidates = list(root.rglob("models"))
    if not candidates: raise FileNotFoundError(…)
    return max(candidates, key=lambda p: p.stat().st_mtime)
```

Picks the newest `models/` directory by mtime — the most recent training
run's checkpoint.

\\newpage

# Package: `evaluation`

## `evaluation/mission_metrics.py`

The six rescue-outcome metrics from the project plan, plus the
Degradation Resilience Ratio (DRR).

```python
@dataclass
class MissionMetrics:
    survivor_recall:      float
    time_to_verification: float  # nan if no confirmations
    false_positive_trips: int
    hazard_exposure:      int
    ugv_travel_cost:      float
    n_steps:              int
```

The six metrics from §17.6 of the project plan. `time_to_verification`
is NaN if no survivors were confirmed (no observations to average) — the
JSON exporter converts NaN → `null` so the web viewer can load the file.

### `class EpisodeRecorder`

Reads scenario state once per env-step and accumulates the bookkeeping
needed to compute MissionMetrics at episode end. Tracks one specific
batch index — for multi-env rollouts, build one recorder per env you
care about.

```python
self.scout_step:   List[int] = [-1] * n_surv  # first-scout step per survivor
self.confirm_step: List[int] = [-1] * n_surv  # first-confirm step
self.travel_cost     = 0.0  # cumulative ground-robot path length
self.hazard_exposure = 0    # ground-robot step-count on burning cells
```

`-1` is the sentinel for "not yet happened". We track first-scout and
first-confirm so we can compute time-to-verification as the gap between
them.

```python
def step(self):
    scouted = self.scenario.scouted_survivors[self.env_index].cpu().tolist()
    found   = self.scenario.found_survivors[self.env_index].cpu().tolist()
    for i in range(self.scenario.n_survivors):
        if scouted[i] and self.scout_step[i] < 0:
            self.scout_step[i] = self.n_steps
        if found[i] and self.confirm_step[i] < 0:
            self.confirm_step[i] = self.n_steps
```

On each step, update the first-event-step for any survivor that just
flipped to scouted/found. Subsequent updates are no-ops because of the
`< 0` guard.

```python
    pos = torch.stack([a.state.pos[b] for a in ground_agents], dim=0)
    if self._prev_ground_pos is not None:
        step_dist = (pos - self._prev_ground_pos).norm(dim=-1).sum().item()
        self.travel_cost += step_dist
    self._prev_ground_pos = pos.clone()
    in_fire = self.scenario._agents_in_fire(ground_agents)[b].cpu().tolist()
    self.hazard_exposure += int(sum(in_fire))
```

Travel cost: per-step ground-robot displacement, summed across robots
and steps. Hazard exposure: count of ground-robot-steps where a robot
stands on a burning cell.

```python
def finalize(self):
    recall = sum(self.scenario.found_survivors[b].cpu().tolist()) / n_surv
    gaps = [self.confirm_step[i] - self.scout_step[i]
            for i in range(n_surv)
            if self.confirm_step[i] >= 0 and self.scout_step[i] >= 0]
    ttv = float(sum(gaps) / len(gaps)) if gaps else float("nan")
    return MissionMetrics(survivor_recall=recall,
                          time_to_verification=ttv, …)
```

End-of-episode aggregation. Note `ttv` is NaN if no survivors were
confirmed (otherwise we'd divide by 0).

### `evaluate_policy(action_fn, …)`

Universal eval entry point. Same harness for baselines and trained
policies — anything with the `policy(env) → action list` signature plugs
in.

```python
def evaluate_policy(n_steps=200, seed=0, num_envs=2,
                   action_fn=None, scenario_kwargs=None, …):
    env = vmas.make_env(scenario=WildfireSearchScenario(),
                        num_envs=num_envs, seed=seed,
                        continuous_actions=True, **scenario_kwargs)
    env.reset()
    if action_fn is None:
        action_fn = lambda env: env.get_random_actions()
    recorder = EpisodeRecorder(env.scenario, env_index=0)
    for _ in range(n_steps):
        env.step(action_fn(env))
        recorder.step()
        if env.scenario.done()[0].item(): break
    return recorder.finalize()
```

### `degradation_resilience_ratio(metrics_by_dropout, metric=…)`

DRR — the custom comms-ablation metric.

```python
def degradation_resilience_ratio(metrics_by_dropout, metric="survivor_recall",
                                 baseline_dropout=0.0):
    base  = getattr(metrics_by_dropout[baseline_dropout], metric)
    worst = getattr(metrics_by_dropout[max(metrics_by_dropout)], metric)
    higher_is_better = metric in {"survivor_recall"}
    ratio = (worst / base) if higher_is_better else (base / max(worst, 1e-9))
    return float(max(0.0, min(1.0, ratio)))
```

`DRR = 1.0` means the metric is unchanged at maximum dropout (perfectly
robust). `DRR = 0.0` means complete collapse. Direction-aware so it works
for both higher-is-better and lower-is-better metrics.

## `evaluation/trajectory_export.py`

Per-step state → JSON dump for the web viewer. The viewer can replay any
trajectory without needing Python at all.

```python
def export_trajectory(strategy_name, make_policy, output_path,
                      n_steps=200, seed=0, num_envs=2, env_index=0,
                      scenario_kwargs=None):
    env = vmas.make_env(scenario=WildfireSearchScenario(),
                        num_envs=num_envs, seed=seed, …)
    env.reset()
    action_fn = make_policy(env)  # policy is built AFTER the env exists
    recorder = EpisodeRecorder(env.scenario, env_index=env_index)
    metadata = {"strategy": strategy_name, "seed": seed, …}
    frames = [{"step": 0, "agents": [_agent_record(a, env_index) for a in env.scenario.world.agents],
              "survivors": _survivor_records(env.scenario, env_index),
              "fire_cells": _fire_cells(env.scenario, env_index)}]
    for step in range(1, n_steps + 1):
        env.step(action_fn(env))
        recorder.step()
        frames.append({…})
        if env.scenario.done()[env_index].item(): break
    metadata["metrics"] = recorder.finalize().as_dict()
    payload = _sanitize_for_json({"metadata": metadata, "frames": frames})
    output_path.write_text(json.dumps(payload, allow_nan=False))
    return output_path
```

The `make_policy(env)` indirection is important: baseline policies bind
to a specific env via constructor (`LawnmowerPolicy(env)` stores
`env.scenario` for later lookup). If we built the policy *before* the env
was built, it would reference the wrong scenario object.

```python
def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_sanitize_for_json(v) for v in obj]
    return obj
```

Recursively converts NaN/±inf → None so `json.dumps(allow_nan=False)`
doesn't blow up and the resulting JSON is strict-parseable in
JavaScript (`JSON.parse` rejects `NaN`).

\\newpage

# Package: `scripts`

Top-level entry points. Mostly thin CLI wrappers around the helper
modules. Worth a brief tour each.

## `scripts/train_mappo_smoke.py`

```python
def build_experiment_config():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = 3
    cfg.on_policy_collected_frames_per_batch = 2_000
    cfg.evaluation = False
    cfg.loggers = []
    cfg.save_folder = str(ROOT / "results")
    return cfg

def main():
    task = make_wildfire_task(max_steps=150)
    experiment = Experiment(task=task,
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        seed=0, config=build_experiment_config())
    experiment.run()
```

3 iters × 2000 frames = 6000 env steps. ~3 s on CPU. The successful exit
is the milestone — the policy is far from converged. For a real run, edit
`build_experiment_config()` (or use `research_config()` from
`train_helpers.py`).

## `scripts/train_ippo_smoke.py`

Same as the MAPPO script but uses `IppoConfig.get_from_yaml()` and the
shared `smoke_config()` from `train_helpers.py`.

## `scripts/train_happo_smoke.py`

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-env-steps", type=int, default=8_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--comms-dropout", type=float, default=0.0)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--exp-name", default="happo_smoke")
    args = p.parse_args()
    result = train_happo(seed=args.seed,
                         num_env_steps=args.num_env_steps,
                         comms_dropout=args.comms_dropout,
                         entropy_coef=args.entropy_coef,
                         n_rollout_threads=8, exp_name=args.exp_name)
    print(f"last mean episode reward: {result['mean_episode_reward']:+.3f}")
```

CLI front-end for `agents.harl_runner.train_happo()`. The flags expose
the knobs that actually matter for HAPPO experimentation —
`num_env_steps`, `entropy_coef`, `comms_dropout`. Everything else stays
default.

## `scripts/comms_dropout_sweep.py`

The headline experiment script. 3 algorithms × 4 dropouts × N seeds,
plus Mann–Whitney U significance tests.

```python
def _run_benchmarl(algo_name, seed, comms_dropout, frames_per_iter, iters):
    cfg = smoke_config(iters=iters, frames_per_batch=frames_per_iter)
    algo_cfg = {"mappo": MappoConfig, "ippo": IppoConfig}[algo_name].get_from_yaml()
    task = make_wildfire_task(comms_dropout=comms_dropout, max_steps=150)
    exp = Experiment(task=task, algorithm_config=algo_cfg,
                     model_config=MlpConfig.get_from_yaml(),
                     seed=seed, config=cfg)
    exp.run()
    return {"metric": float(exp.mean_return), "wall_sec": …}

def _run_happo(seed, comms_dropout, num_env_steps):
    r = train_happo(seed=seed, num_env_steps=num_env_steps,
                    comms_dropout=comms_dropout, n_rollout_threads=8,
                    exp_name=f"happo_d{int(comms_dropout*100)}_s{seed}")
    return {"metric": r["mean_episode_reward"], "wall_sec": r["wall_sec"]}
```

Two dispatch functions, one per library. The "metric" returned is the
algorithm's native training-time mean return for BenchMARL,
mean_episode_reward for HARL. The values aren't directly comparable
*between* algorithms in absolute terms — but within one algorithm they
*are* comparable across dropouts, which is what the sweep actually
needs.

```python
def mann_whitney(x, y):
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return float(u), float(p)
```

Per algorithm, per "d=0 vs higher dropout" pair, we test whether the
distributions differ. At N=3 vs N=3 the minimum two-sided p is 0.10, so
the smoke budget can never produce p<0.05 — flag this clearly in the
script's docstring and the README.

```python
def main():
    # parse args
    # for each (algo, dropout, seed): run_cell(); append to cells
    # aggregate: mean ± std per (algo, dropout)
    # significance: for each algo, MW-U of (d=base) vs (d=other)
    # print summary table + significance table
    # dump cells + summary + tests to results/comms_dropout_sweep_*.json
```

Output is a structured JSON the viewer notebook (now removed) and the
README results table both consume.

## `scripts/compare_baselines.py`

```python
def run_one(strategy_name, seed, steps):
    env = vmas.make_env(scenario=WildfireSearchScenario(),
                        num_envs=2, seed=seed, …)
    env.reset()
    cls = BASELINES[strategy_name]
    policy = cls() if cls is RandomActionPolicy else cls(env)
    rec = EpisodeRecorder(env.scenario, env_index=0)
    for _ in range(steps):
        env.step(policy(env))
        rec.step()
        if env.scenario.done()[0].item(): break
    return rec.finalize()
```

For each (strategy, seed): build the env, build the policy, run the
episode, score. The mission metrics from `evaluation/mission_metrics.py`
are the universal currency — same struct returned for every strategy.

```python
def main():
    # for each strategy:
    #   for each seed:
    #     per_seed.append(run_one(...))
    #   compute mean/std on each metric, print row
    # dump to results/baseline_comparison_*.json
```

The output JSON is the input to notebook 03 (baseline comparison plots).

## `scripts/export_trajectories.py`

```python
def main():
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--comms-dropout", type=float, default=0.3)
    args = p.parse_args()
    for name, cls in BASELINES.items():
        def make_policy(env, _cls=cls):
            return _cls() if _cls is RandomActionPolicy else _cls(env)
        export_trajectory(strategy_name=name, make_policy=make_policy,
                          output_path=out_dir / f"{name}.json",
                          n_steps=args.steps, seed=args.seed,
                          scenario_kwargs={"comms_dropout": args.comms_dropout})

    # Also export the trained HAPPO if a checkpoint exists
    try:
        ckpt = find_latest_happo_checkpoint().resolve()
        def make_happo(env, _ckpt=ckpt):
            return HappoPolicy.from_checkpoint(_ckpt)
        export_trajectory(strategy_name="happo_trained",
                          make_policy=make_happo, …)
    except FileNotFoundError:
        print("  ⚠ HAPPO export skipped — run train_happo_smoke.py first.")
```

Produces `web/trajectories/*.json` — the inputs the viewer reads. Each
JSON is `~200 KB`, one per strategy, 5 strategies total when HAPPO has
been trained.

\\newpage

# Package: `web`

The browser viewer. A single self-contained `index.html` that loads React
and Three.js from a CDN (no build step) and replays the trajectory JSON
files written by `export_trajectories.py`.

## `web/index.html`

One file with three layers: a `<style>` block (CSS theme + layout), an
import map, and an ES-module `<script>` containing the whole React + Three.js
app. We walk all three.

### The CSS theme

```css
:root {
  --bg:     #0e1118;   --panel:  #151a23;   --border: #1f2735;
  --text:   #e6e9ef;   --muted:  #8b94a8;   --accent: #5d82fa;
  --good:   #88ff88;   --warn:   #ffaa00;   --danger: #ff4444;
}
```

CSS custom properties define the dark palette once at `:root`; every rule
below references them via `var(--name)`. Changing the theme is a one-line
edit. `--accent` (blue) is reused for focus/hover states throughout.

```css
#root { display: flex; flex-direction: column; height: 100vh; }
main  { flex: 1; display: flex; min-height: 0; }
aside { width: 280px; border-left: 1px solid var(--border); }
```

The layout is a vertical flexbox: `header` (fixed), `main` (grows to fill),
`footer` (fixed). Inside `main`, a horizontal flexbox puts the scene canvas
on the left (`flex: 1`) and the 280-px metrics sidebar on the right. The
`min-height: 0` on `main` is the standard flexbox trick that lets the canvas
child shrink instead of overflowing.

The remaining CSS styles the legend overlay (absolute-positioned top-left of
the scene), the collapsible legend toggle, swatch shapes (`.circle`,
`.square`, `.tri` — the triangle is drawn with the classic transparent-border
trick), the metric rows, and the footer playback controls. It is all
presentational; nothing here carries logic.

### The import map

```html
<script type="importmap">
{ "imports": {
    "three":     "https://esm.sh/three@0.171.0",
    "react":     "https://esm.sh/react@18.3.1",
    "react-dom": "https://esm.sh/react-dom@18.3.1",
    "react-dom/client": "https://esm.sh/react-dom@18.3.1/client"
} }
</script>
```

An import map lets the module script below write bare specifiers like
`import * as THREE from 'three'` and have the browser resolve them to pinned
CDN URLs. This is what removes the build step: no bundler, no `node_modules`.
Versions are pinned (`@0.171.0`, `@18.3.1`) so the demo can't break when a CDN
ships a new major.

### Module setup

```js
import * as THREE from 'three';
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { createRoot } from 'react-dom/client';

const h = React.createElement;
```

`h` is the React `createElement` shorthand — because there is no JSX
transpiler, the whole UI is written as `h(tag, props, ...children)` calls.

```js
const STRATEGY_NAMES = ['random_action', 'random_walk', 'lawnmower', 'nearest_candidate',
                        'highest_confidence', 'happo_trained'];

async function loadTrajectory(name) {
  const res = await fetch(`./trajectories/${name}.json`);
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return await res.json();
}
```

The known strategy list (matches the keys exported by
`export_trajectories.py`) and a tiny fetch helper. A missing file throws,
which the `App` component catches and turns into an on-screen hint to run the
exporter.

### `setupScene(canvas, width, height)`

```js
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(width, height, false);
renderer.setClearColor(0x0e1118, 1);

const baseHalf = 1.05;
const aspect = width / height;
const halfW = aspect >= 1 ? baseHalf * aspect : baseHalf;
const halfH = aspect >= 1 ? baseHalf            : baseHalf / aspect;
const camera = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.1, 100);
camera.position.set(0, 0, 10);
camera.lookAt(0, 0, 0);
```

Builds the WebGL renderer and an **orthographic** camera (no perspective — a
true top-down map view). The world is `[-1, 1]²`; `baseHalf = 1.05` shows it
with ~5 % margin. The `aspect`-aware half-width/height math keeps the world
square and fully visible regardless of canvas shape: the longer screen axis
gets the extra room, the world never stretches. A dark-green background plane
is added at `z = -0.5` as the "forest floor".

### `clearGroup(group)`

```js
while (group.children.length) {
  const obj = group.children.pop();
  if (obj.geometry) obj.geometry.dispose();
  if (obj.material) obj.material.dispose();
}
```

Three.js doesn't free GPU memory automatically. Because we rebuild the scene
every frame (see `renderFrame`), we must `dispose()` the geometry and material
of every removed mesh or the page leaks VRAM over a long replay.

### The colour palette and entity factories

```js
const COLORS = { drone: 0x4a90ff, ground: 0x14b8a6,
  survivor_unscouted: 0xff3838, survivor_scouted: 0xfbbf24,
  survivor_confirmed: 0x84cc16, fire_outer: 0xf06e1e, ... };
```

A central hex-colour map mirroring the scenario's semantics: blue drones,
teal ground robots, and the red → amber → lime survivor-state progression
(unscouted → scouted → confirmed).

`makeDroneMesh()`, `makeGroundMesh()`, `makeSurvivorMesh(state)` and
`makeFireMesh(size)` each return a Three.js `Group` of primitive meshes
assembled to look like the thing from above:

```js
function makeDroneMesh() {
  const g = new THREE.Group();
  // X-shaped frame, central body, four propellers
  for (const [dx, dy] of [[0.05,0],[-0.05,0],[0,0.05],[0,-0.05]]) {
    const prop = new THREE.Mesh(new THREE.PlaneGeometry(0.032, 0.005), propMat.clone());
    prop.position.set(dx, dy, 0.03);
    prop.userData.isProp = true;             // tag for the animation loop
    prop.userData.phase  = Math.random() * Math.PI;
    g.add(prop);
  }
  return g;
}
```

The pattern worth noting: meshes that animate are tagged via
`userData` (`isProp`, `isFireLayer`, `isPulse`, `isCommsDot`). The per-frame
animation loop (`tickAnimation`) traverses the groups and updates only the
tagged meshes — so propeller spin and fire flicker happen without rebuilding
geometry. Each tagged mesh also stores a random `phase` so the drones'
propellers and the fire cells don't all flicker in lockstep.

`makeSurvivorMesh` colours the figure (head, torso, arms, legs, plus a state
ring) by `state.found ? confirmed : state.scouted ? scouted : unscouted`, and
tags the ring `isPulse` when scouted-but-not-confirmed so it gently pulses —
a visual call-for-attention on survivors awaiting a ground robot.

`makeFireMesh` stacks five concentric circles (dark-red base → white-hot
core), each tagged `isFireLayer` with a per-layer `wobble` amount so outer
layers dance more than the core.

### `renderFrame(state, frame, meta, histories)`

The heart of the viewer — rebuilds the whole scene for one trajectory frame.

```js
clearGroup(fireGroup); clearGroup(agentGroup);
clearGroup(survGroup); clearGroup(trailGroup);

const G = meta.fire_grid_size;
const cell = 2 / G;
for (const [gx, gy] of frame.fire_cells) {
  const mesh = makeFireMesh(cell);
  mesh.position.set(-1 + (gx + 0.5) * cell, -1 + (gy + 0.5) * cell, -0.3);
  fireGroup.add(mesh);
}
```

First clears the four scene groups, then re-adds fire cells. The fire grid is
discrete (`G×G`); `cell = 2/G` is one cell's width in world units, and the
`-1 + (g + 0.5) * cell` maps a grid index to the centre of that cell in
`[-1, 1]` space.

```js
for (const a of frame.agents) {
  const hist = (histories[a.name] || []).slice(
    Math.max(0, frame.step - TRAIL_LEN), frame.step + 1);
  if (hist.length < 2) continue;
  const positions = new Float32Array(hist.length * 3);
  // fill x,y,z=-0.04 for each historical point
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  trailGroup.add(new THREE.Line(geom, mat));
}
```

Each agent gets a fading trail: the last `TRAIL_LEN = 25` positions, pulled
from the pre-computed `histories` map, drawn as a `THREE.Line`. Coloured by
agent type (blue/teal).

```js
for (const a of frame.agents) {
  const mesh = a.type === 'drone' ? makeDroneMesh() : makeGroundMesh();
  mesh.position.set(a.x, a.y, 0.05);
  if (a.type === 'ground' && histories[a.name]) {
    const [px, py] = histories[a.name][frame.step - 1];
    if (Math.abs(a.x-px) + Math.abs(a.y-py) > 1e-4)
      mesh.rotation.z = Math.atan2(a.y - py, a.x - px);  // face travel direction
  }
  agentGroup.add(mesh);
  const commsUp = a.comms_up !== false;
  const commsDot = new THREE.Mesh(new THREE.CircleGeometry(0.013, 16),
    new THREE.MeshBasicMaterial({ color: commsUp ? 0x4ade80 : 0xef4444 }));
  commsDot.position.set(a.x, a.y + 0.062, 0.09);
  commsDot.userData.isCommsDot = true;
  commsDot.userData.isUp = commsUp;
  agentGroup.add(commsDot);
}
```

Agents are placed at their frame position. Ground robots are rotated to face
their direction of travel (computed from the previous step's position). Above
each agent sits a small **comms dot**: green when `comms_up` is truthy, red
when the radio dropped this step (`a.comms_up === false`). This is the direct
visual readout of the `comms_dropout` model — the field comes from
`scenario._neighbor_observations` and is written per-frame by
`trajectory_export.py`. Older JSONs without the field default to green
(`!== false`).

### `tickAnimation(state, t)`

```js
state.agentGroup.traverse(obj => {
  if (obj.userData?.isProp) obj.rotation.z = t * 18 + (obj.userData.phase || 0);
});
state.fireGroup.traverse(obj => {
  if (obj.userData?.isFireLayer) {
    obj.material.opacity = Math.max(0.05, baseOp * (0.7 + 0.3*Math.sin(t*8 + phase)));
    const s = 1 + wobble * Math.sin(t*11 + phase*1.7);
    obj.scale.set(s, s, 1);
  }
});
```

Runs on every animation frame with `t` = seconds since load. It spins
propellers, flickers fire opacity/scale, pulses scouted-survivor rings, and
pulses the comms dot when it's *down*. Note it mutates only `userData`-tagged
meshes — independent of which trajectory frame is showing, so the scene feels
alive even when playback is paused.

### `buildHistories(frames)`

```js
function buildHistories(frames) {
  const out = {};
  for (const f of frames)
    for (const a of f.agents) (out[a.name] ??= []).push([a.x, a.y]);
  return out;
}
```

Pre-computes, once per loaded trajectory, the full `[x, y]` position history
per agent. `renderFrame` slices this for trails and `atan2` heading — cheaper
than scanning all frames every render.

### `useThree(meta, frame, histories)` — the React/Three.js bridge

```js
useEffect(() => {                       // runs once (empty deps)
  const { renderer, scene, camera } = setupScene(canvas, rect.width, rect.height);
  // create fireGroup / trailGroup / survGroup / agentGroup, add to scene
  const tick = () => {
    const t = (performance.now() - t0) / 1000;
    tickAnimation(stateRef.current, t);
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  };
  tick();
  window.addEventListener('resize', onResize);
  return () => { cancelAnimationFrame(raf); /* …cleanup… */ };
}, []);

useEffect(() => {                       // runs when frame/meta/histories change
  renderFrame(stateRef.current, frame, meta, histories);
}, [meta, frame, histories]);
```

A custom hook that owns the Three.js lifecycle. The first `useEffect` (empty
dependency array) sets up the renderer, scene groups, the `requestAnimationFrame`
render loop, and a resize handler — and returns a cleanup that cancels the RAF
and disposes the renderer when the component unmounts. The second `useEffect`
calls `renderFrame` whenever the current frame changes. This cleanly separates
the *continuous* animation (RAF loop) from the *discrete* per-frame scene
rebuild (React effect).

### `usePlayback(nFrames)`

```js
const [step, setStep] = useState(0);
const [playing, setPlaying] = useState(true);
const [speed, setSpeed] = useState(1);
useEffect(() => {
  if (!playing) return;
  const id = setInterval(() => setStep(s => {
    const next = s + speed;
    if (next >= nFrames) { setPlaying(false); return nFrames - 1; }
    return next;
  }), 100);
  return () => clearInterval(id);
}, [playing, speed, nFrames]);
```

The playback clock: a `setInterval` at 10 Hz advancing `step` by `speed`
frames per tick (so the 1×/2×/4×/8× selector just changes the stride). It
stops and parks at the last frame when the trajectory ends.

### Components: `Legend`, `MetricsPanel`, `App`

`Legend` is a collapsible overlay (`useState(open)`) that renders the
shape/colour key and an explanatory note, all via `h(...)` calls.

`MetricsPanel({ meta })` shows the strategy name, team composition, and the
six mission metrics from `meta.metrics`, formatting `null`/`NaN` as an em-dash
(`—`) and rounding floats to two decimals.

`App` is the root:

```js
function App() {
  const [strategy, setStrategy] = useState(STRATEGY_NAMES[2]); // nearest_candidate
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    setData(null); setError(null);
    loadTrajectory(strategy).then(setData).catch(e => setError(e.message));
  }, [strategy]);

  const frames = data?.frames ?? [];
  const histories = useMemo(() => buildHistories(frames), [data]);
  const { step, setStep, playing, setPlaying, speed, setSpeed } = usePlayback(frames.length);
  const frame = frames[step] ?? null;
  const canvasRef = useThree(data?.metadata, frame, histories);
  return h(React.Fragment, null, /* header, main(scene+panel), footer */);
}

createRoot(document.getElementById('root')).render(h(App));
```

`App` wires everything together: it fetches the selected strategy's
trajectory, recomputes histories only when the data changes (`useMemo`), runs
the playback clock, and feeds the current frame into `useThree`. The header
holds the strategy `<select>`; the footer holds restart/play/pause, the
scrubber `<input type=range>`, and the speed selector. The final line mounts
`App` into `#root`. The whole interactive viewer is a few hundred lines with
zero build tooling.

\\newpage

# Notebooks

Three Jupyter notebooks under `notebooks/`, each verifying one layer of the
stack. They are thin: they load `results/*.json` artifacts and render
tables/plots, so the heavy lifting stays in the scripts. Documented in full
in `notebooks/README.md`.

## `notebooks/01_setup_and_demo.ipynb`

Environment and dependency verification (~5 s). Sections: a Python/platform
check; an optional (commented-out) dependency install cell; per-component
import checks for PyTorch, VMAS, TorchRL, BenchMARL and W&B (each prints a `✓`
or raises so a broken dependency is obvious); a hello-world VMAS demo running
the built-in `navigation` scenario for 100 random steps across 32 parallel
envs; and a markdown preview of what a BenchMARL training call looks like.
Run this first — it tells you exactly which dependency to fix if anything is
missing.

## `notebooks/02_sweep_results.ipynb`

Loads the most recent `results/comms_dropout_sweep_*.json` (written by
`scripts/comms_dropout_sweep.py`) and visualises it as: a pivot table of
mean ± std per (algorithm × dropout) cell, a line plot of mean return vs
`comms_dropout` (one line per algorithm), and a wall-time bar chart sanity
check. It re-discovers the newest sweep file every run, so the loop is:
re-run the sweep, re-run the notebook. The README flags the key caveat — at
smoke budget the curves are noisy, and BenchMARL's `mean_return` vs HARL's
`mean_episode_reward` are only comparable *within* an algorithm.

## `notebooks/03_baseline_comparison.ipynb`

Loads the most recent `results/baseline_comparison_*.json` (written by
`scripts/compare_baselines.py`) and renders a mean ± std table, a 2×2 bar
panel (one chart per numeric mission metric), and a per-metric winners table.
This is the headline "does any heuristic beat the others?" output;
`nearest_candidate` tends to win on recall at the current scenario config.
Trained MAPPO/IPPO/HAPPO policies will slot into the same plots once their
checkpoint loaders are wired into `compare_baselines.py`.

\\newpage

# Closing notes

This walkthrough now covers every source file the project ships: the VMAS
scenario (`envs/`), the library adapters and trained-policy loader
(`agents/`), the metrics and exporters (`evaluation/`), the CLI entry points
(`scripts/`), the browser viewer (`web/index.html`), and the three notebooks.

The shape of the whole system, restated in one breath: the scenario is
defined once in `envs/wildfire_search.py`; BenchMARL (MAPPO/IPPO) and HARL
(HAPPO) train on it through thin adapters; `evaluation/` scores rollouts on
the same six mission metrics regardless of policy; and `web/index.html`
replays the exported trajectories in the browser. Everything else is glue
reshaping VMAS's batched tensors into whatever each downstream library wants.

If anything in the code is unclear, the README has a TL;DR run-everything
block; the per-script docstrings describe what each entry point does; and the
project plan PDF documents the *why* behind each design choice. This document
itself is generated by `docs/build_walkthrough.py` — edit the walkthrough
string there and re-run to regenerate the PDF.
"""


# ----------------------------------------------------------------------
# Fonts — DejaVu (shipped with matplotlib) gives full Unicode coverage
# (box-drawing, arrows, ✓, ±, ×) that ReportLab's built-in Type-1 fonts
# lack. Falls back to Helvetica/Courier if DejaVu isn't importable.
# ----------------------------------------------------------------------
BODY_FONT = "Helvetica"
BODY_BOLD = "Helvetica-Bold"
BODY_ITALIC = "Helvetica-Oblique"
MONO_FONT = "Courier"
MONO_BOLD = "Courier-Bold"


def _register_fonts() -> None:
    global BODY_FONT, BODY_BOLD, BODY_ITALIC, MONO_FONT, MONO_BOLD
    try:
        import matplotlib
    except ImportError:
        return
    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    fonts = {
        "DejaVuSans": ttf / "DejaVuSans.ttf",
        "DejaVuSans-Bold": ttf / "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique": ttf / "DejaVuSans-Oblique.ttf",
        "DejaVuSansMono": ttf / "DejaVuSansMono.ttf",
        "DejaVuSansMono-Bold": ttf / "DejaVuSansMono-Bold.ttf",
    }
    if not all(p.exists() for p in fonts.values()):
        return
    for name, path in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
        italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-Bold",
    )
    BODY_FONT, BODY_BOLD, BODY_ITALIC = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique"
    MONO_FONT, MONO_BOLD = "DejaVuSansMono", "DejaVuSansMono-Bold"


# ----------------------------------------------------------------------
# Inline + code formatting helpers
# ----------------------------------------------------------------------
TEAL = colors.HexColor("#14b8a6")
NAVY = colors.HexColor("#1a3a8a")
DEEP_TEAL = colors.HexColor("#0f4f48")
INK = colors.HexColor("#1a1f2c")
CODE_BG = colors.HexColor("#f5f7fa")
CODE_BORDER = colors.HexColor("#d8dce5")
INLINE_BG = colors.HexColor("#eef1f6")

_LEXERS = {
    "python": PythonLexer, "py": PythonLexer,
    "javascript": JavascriptLexer, "js": JavascriptLexer, "jsx": JavascriptLexer,
    "html": HtmlLexer,
    "css": HtmlLexer,
}
_PYG_STYLE = get_style_by_name("default")


def _inline(text: str) -> str:
    """Convert a markdown-ish span to ReportLab mini-markup.

    Handles escaping, **bold**, `code`, and *italic*.
    """
    text = html.escape(text, quote=False)
    # Stash inline-code spans behind placeholders so their contents (which may
    # contain * or other markup chars) are not touched by the bold/italic passes.
    spans: list[str] = []

    def _stash(m: "re.Match") -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    def _restore(m: "re.Match") -> str:
        code = spans[int(m.group(1))]
        return f'<font face="{MONO_FONT}" size="9" color="#b5076b">{code}</font>'

    return re.sub(r"\x00(\d+)\x00", _restore, text)


def _highlight(code: str, lang: str) -> str:
    """Pygments-highlight `code` into ReportLab XPreformatted markup."""
    lexer_cls = _LEXERS.get((lang or "").lower())
    if lexer_cls is None:
        return html.escape(code, quote=False)
    out = []
    for ttype, value in lex(code, lexer_cls()):
        esc = html.escape(value, quote=False)
        st = _PYG_STYLE.style_for_token(ttype)
        prefix, suffix = "", ""
        if st.get("color"):
            prefix += f'<font color="#{st["color"]}">'
            suffix = "</font>" + suffix
        if st.get("bold"):
            prefix += "<b>"
            suffix = "</b>" + suffix
        if st.get("italic"):
            prefix += "<i>"
            suffix = "</i>" + suffix
        out.append(prefix + esc + suffix)
    return "".join(out)


# ----------------------------------------------------------------------
# Paragraph styles
# ----------------------------------------------------------------------
def _styles() -> dict:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle(
        "Body", parent=base, fontName=BODY_FONT, fontSize=10.5, leading=15.5,
        textColor=INK, spaceBefore=2, spaceAfter=7, alignment=TA_LEFT,
    )
    return {
        "body": body,
        "list": ParagraphStyle("List", parent=body, leftIndent=16, spaceAfter=4),
        "h1": ParagraphStyle("H1", parent=body, fontName=BODY_BOLD, fontSize=21,
                             leading=25, textColor=colors.HexColor("#0e1118"),
                             spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("H2", parent=body, fontName=BODY_BOLD, fontSize=15,
                             leading=19, textColor=NAVY, spaceBefore=16, spaceAfter=3),
        "h3": ParagraphStyle("H3", parent=body, fontName=BODY_BOLD, fontSize=12,
                             leading=15, textColor=DEEP_TEAL, spaceBefore=12, spaceAfter=2),
        "h4": ParagraphStyle("H4", parent=body, fontName=BODY_BOLD, fontSize=10.8,
                             leading=14, textColor=colors.HexColor("#444444"),
                             spaceBefore=9, spaceAfter=1),
        "code": ParagraphStyle("Code", parent=base, fontName=MONO_FONT, fontSize=8.0,
                               leading=10.8, textColor=INK, backColor=CODE_BG,
                               borderColor=CODE_BORDER, borderWidth=0.5,
                               borderPadding=(6, 6, 6, 6), leftIndent=2, rightIndent=2,
                               spaceBefore=4, spaceAfter=9,
                               wordWrap="CJK", splitLongWords=1,
                               allowWidows=1, allowOrphans=1),
    }


# ----------------------------------------------------------------------
# Markdown → flowables
# ----------------------------------------------------------------------
def _is_newpage(line: str) -> bool:
    return line.strip().lstrip("\\") == "newpage"


def _build_flowables(md: str, styles: dict) -> list:
    flow: list = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    para: list[str] = []

    def flush_para() -> None:
        if not para:
            return
        text = " ".join(s.strip() for s in para).strip()
        para.clear()
        if not text:
            return
        # List item?
        m = re.match(r"^(\d+\.|[-*])\s+(.*)$", text)
        if m:
            bullet = "•" if m.group(1) in ("-", "*") else m.group(1)
            flow.append(Paragraph(f"{bullet}&nbsp;&nbsp;{_inline(m.group(2))}",
                                  styles["list"]))
        else:
            flow.append(Paragraph(_inline(text), styles["body"]))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if _is_newpage(line):
            flush_para()
            flow.append(PageBreak())
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            i += 1
            block: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = "\n".join(block)
            flow.append(XPreformatted(_highlight(code, lang), styles["code"]))
            continue

        if stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            key = {1: "h1", 2: "h2", 3: "h3"}.get(level, "h4")
            flow.append(Paragraph(_inline(text), styles[key]))
            if level == 1:
                flow.append(HRFlowable(width="100%", thickness=2, color=TEAL,
                                       spaceBefore=2, spaceAfter=8))
            elif level == 2:
                flow.append(HRFlowable(width="100%", thickness=0.6,
                                       color=colors.HexColor("#c8d0e0"),
                                       spaceBefore=1, spaceAfter=6))
            i += 1
            continue

        if stripped == "":
            flush_para()
            i += 1
            continue

        # A new list item starts its own paragraph even without a blank line
        # before it, so consecutive "1. … 2. …" items don't get merged.
        if re.match(r"^\s*(\d+\.|[-*])\s+", line) and para:
            flush_para()

        para.append(line)
        i += 1

    flush_para()
    return flow


# ----------------------------------------------------------------------
# Page furniture: running header/footer with "Page X / Y"
# ----------------------------------------------------------------------
class WalkthroughDoc(BaseDocTemplate):
    def __init__(self, path: str, **kw):
        super().__init__(path, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                            onPage=self._decorate)])
        self._page_count = 0

    def _decorate(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(BODY_FONT, 8.5)
        canvas.setFillColor(colors.HexColor("#8b94a8"))
        canvas.drawString(doc.leftMargin, 1.1 * cm, "OmniSearch — Code Walkthrough")
        canvas.drawRightString(doc.leftMargin + doc.width, 1.1 * cm,
                               f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#d8dce5"))
        canvas.line(doc.leftMargin, 1.45 * cm,
                    doc.leftMargin + doc.width, 1.45 * cm)
        canvas.restoreState()


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def main() -> None:
    _register_fonts()
    styles = _styles()

    # Title block + the walkthrough body.
    flow: list = [
        Paragraph("OmniSearch", ParagraphStyle(
            "Title", fontName=BODY_BOLD, fontSize=30, leading=34,
            textColor=colors.HexColor("#0e1118"), spaceAfter=2)),
        Paragraph("Code Walkthrough", ParagraphStyle(
            "Sub", fontName=BODY_FONT, fontSize=16, leading=20,
            textColor=TEAL, spaceAfter=10)),
        HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=12),
    ]
    flow += _build_flowables(WALKTHROUGH, styles)

    out_pdf = ROOT / "docs" / "code_walkthrough.pdf"
    doc = WalkthroughDoc(
        str(out_pdf), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="OmniSearch — Code Walkthrough", author="OmniSearch",
    )
    doc.build(flow)
    print(f"wrote {out_pdf.relative_to(ROOT)}  ({out_pdf.stat().st_size:_} bytes)")


if __name__ == "__main__":
    main()

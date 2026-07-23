# Baseline Approaches and Route Planning

This document describes the hand-coded comparison policies used in OmniSearch
and the route-planning logic behind their UGV behavior. The goal is to make the
baselines interpretable: each one tests a specific non-learning strategy under
the same simulator, perception model, action space, and evaluation metrics as
the learned HAPPO policy.

The baselines are implemented in `agents/baselines.py`. Each policy is a
callable controller with the same interface as a trained policy:

```text
policy(env) -> list of per-agent continuous 2D actions
```

This means that the comparison does not change the simulator physics. UAVs and
UGVs still move through the same terrain, observe survivors through the same
perception model, and are evaluated through the same mission metrics. What
changes is only how actions are selected.

## 1. What the Baselines Are Designed to Test

The baselines are not intended to be weak strawman controllers. They separate
several capabilities that a learned multi-agent policy should ideally combine:

| Baseline | UAV behavior | UGV behavior | Main question tested |
|---|---|---|---|
| `random_action` | independent random actions | random until a survivor is locally known, then route to a target | How much performance comes from chance discovery alone? |
| `random_walk` | persistent random headings | persistent random walk until a target is known, then route to a target | Does simple spatial persistence improve over uncorrelated random motion? |
| `lawnmower` | land-aware serpentine coverage | route to locally known, unconfirmed survivors | How strong is a structured coverage pattern without learning? |
| `highest_confidence` | lawnmower coverage | prioritize the strongest retained detection confidence | Does confidence-based target prioritization improve confirmation? |
| `ant_colony` | move toward least-recently-seen searchable cells | route using locally known survivor memory | Can decentralized recency memory approximate adaptive exploration? |
| `matched_heuristic` | lawnmower coverage | follow the same scenario assignment and planner hint used by HAPPO | How much comes from planner-aware UGV routing rather than learned UAV search? |

All baselines operate with the same role split as the learned system: UAVs scout
from the air, and UGVs confirm survivors on the ground after a survivor becomes
known through mission memory.

## 2. UAV Search Baselines

### Random Action

`random_action` is the lower-bound search policy. UAV actions are sampled from
the environment's random action generator. UGVs also use random actions until a
survivor is locally known; after that, they switch to the shared UGV target
assignment and route-planning logic.

Conceptually, this baseline answers whether the environment is so easy that
unstructured motion can solve it. In realistic terrain and short episodes, it is
expected to waste substantial time revisiting already inspected areas or moving
away from useful search regions.

### Persistent Random Walk

`random_walk` keeps a heading for each agent and perturbs it gradually instead
of resampling an unrelated action every step. In the current implementation,
the heading diffusion uses a persistence time of `20 s`, and the action
magnitude is `0.95`. When a proposed step would leave the world boundary, the
heading is reflected.

This is still uninformed exploration, but it avoids the jittery behavior of
pure random actions. It is useful as a baseline for testing whether a learned
policy is doing more than simply moving smoothly through the map.

### Lawnmower Coverage

`lawnmower` is the main structured coverage baseline. UAVs follow
boustrophedon, or serpentine, lanes across the searchable part of the map.
Water and rock cells are excluded from the search workload, and each lane is
trimmed to the land segment it covers. Lanes are then divided across drones by
searchable workload rather than by raw geometric width, so agents receive more
balanced coverage assignments on irregular terrain.

The lane spacing is derived from the UAV camera footprint. The implementation
uses

```text
lane spacing = 1.2 * camera radius
```

which corresponds to approximately `0.6` times the full camera footprint width.
UAVs brake near lane endpoints with a waypoint tolerance of `5 m`, a slowdown
distance of `40 m`, and an arrival damping factor of `0.65`. This makes the
coverage pattern physically smoother and avoids excessive overshoot at lane
turns.

### Highest-Confidence Targeting

`highest_confidence` keeps the same UAV lawnmower search pattern, but changes
how UGVs choose among known survivors. Instead of simply assigning the nearest
unconfirmed target, the policy retains the strongest detection confidence seen
for each survivor and prioritizes higher-confidence targets first. Each selected
target is then assigned to the nearest available UGV.

This baseline isolates the value of probabilistic perception for the
confirmation stage. If it outperforms nearest-target routing, then detection
confidence is useful not only as a diagnostic but also as a decision signal.

### Ant-Colony Recency Search

`ant_colony` is a decentralized coverage heuristic inspired by stigmergic
search. Each UAV maintains a timestamp map over grid cells. When the UAV
observes a cell inside its camera footprint, that cell receives the current
step index. The UAV then chooses a new target among the least-recently-seen
searchable cells, preferring the nearest such cell.

Let $C$ be the set of grid cells, $S \subset C$ the searchable cells, and
$T_{i,t}(c)$ the last time UAV $i$ observed cell $c$. Unseen cells are
initialized with a sentinel value $T_{i,t}(c)=-1$. For UAV position $p_{i,t}$
and camera footprint radius $r_{i,t}$, the observed cells are

$$
O_{i,t}
=
\left\{
c \in C :
\left\lVert x(c)-p_{i,t}\right\rVert_2 \le r_{i,t}
\right\},
$$

where $x(c)$ is the world-coordinate center of cell $c$. The local pheromone
timestamp update is

$$
T_{i,t+1}(c)
=
\begin{cases}
t, & c \in O_{i,t},\\
T_{i,t}(c), & \text{otherwise}.
\end{cases}
$$

Connected UAVs merge their maps by keeping the newest timestamp:

$$
T_{i,t+1}(c)
=
\max_{j \in \mathcal{N}_{i,t}} T_{j,t+1}(c),
$$

where $\mathcal{N}_{i,t}$ is the set of drones connected to UAV $i$ at that
step. During communication dropout, the local map is not overwritten by the team
map.

The implementation does not sum a literal continuous repulsive force from all
visited cells. Instead, recent observations act as a discrete repulsive memory:
fresh cells are avoided by selecting from the oldest timestamp level first. The
target cell is

$$
C^\star_{i,t}
=
\operatorname*{arg\,min}_{c \in S} T_{i,t}(c),
$$

followed by a nearest-cell tie break:

$$
c^\star_{i,t}
=
\operatorname*{arg\,min}_{c \in C^\star_{i,t}}
\left\lVert x(c)-p_{i,t}\right\rVert_2^2 .
$$

This is equivalent to a strong recency repulsion: recently observed cells have
higher timestamps and are therefore excluded until older or unseen cells have
been revisited. After selecting $c^\star_{i,t}$, the UAV moves toward its cell
center using the same damped arrival controller as the lawnmower baseline.

The behavior is "ant-colony-like" because it avoids recently visited regions
without requiring a global precomputed sweep. Connected agents merge their most
recent timestamp maps and survivor events; disconnected agents keep their local
memory until communication returns. This makes the policy a useful comparison
for decentralized adaptive exploration.

## 3. UGV Target Assignment

Most baselines share the same basic UGV logic:

1. A survivor must be locally known before a UGV can target it.
2. Already confirmed survivors are excluded.
3. Available UGVs are assigned to distinct targets when possible.
4. The selected target is converted into a feasible ground waypoint.
5. The waypoint is converted into a continuous action in `[-1, 1]^2`.

The standard assignment is nearest-target matching: each UGV selects the
nearest unassigned locally known survivor. The highest-confidence variant first
orders targets by retained detector confidence and then assigns each target to
the nearest available UGV.

## 4. Routing and Pathfinding

UGV movement is terrain constrained. A direct vector toward a survivor can be
invalid if it crosses water, rock, buildings or trees, steep terrain, or a fire
region with high traversal cost. For that reason, the baselines include a
route-aware waypoint layer before the final continuous action is produced.

The native baseline route planner works as follows.

First, the controller checks whether the straight segment from the UGV to the
target is traversable. If it is, the survivor position itself becomes the
waypoint. If the direct path is blocked, both the current UGV position and the
target position are converted to grid cells and snapped to the nearest
traversable cell if needed.

Second, the planner runs weighted A* on the traversability grid using
`agents/pathfinding.py`. The graph is eight-connected by default, so diagonal
motions are allowed, but a diagonal edge is rejected if it would squeeze between
two blocked axial neighbors. This prevents routes from cutting unrealistically
through obstacle corners.

The route cost is based on the terrain mobility grid. In the native baseline
UGV router, fire is added as an extra cost:

$$
\kappa(c)
=
\kappa_{\mathrm{mobility}}(c)
+
25.0\, f(c),
$$

where $\kappa(c)$ is the A* traversal cost of cell $c$,
$\kappa_{\mathrm{mobility}}(c)$ is the terrain-dependent mobility cost, and
$f(c)$ is the fire intensity in that cell.

For an edge $(u,v)$, the implemented cost averages the two endpoint cell costs
and multiplies by the grid step length:

$$
w(u,v)
=
d(u,v)\,
\frac{\kappa(u)+\kappa(v)}{2},
$$

with $d(u,v)=1$ for axial moves and $d(u,v)=\sqrt{2}$ for diagonal moves. A*
then searches for the minimum-cost route

$$
P^\star
=
\operatorname*{arg\,min}_{P:start\rightarrow goal}
\sum_{(u,v)\in P} w(u,v).
$$

This does not make fire absolutely impassable, but it strongly discourages
routes through burning cells when an alternative exists. The A* heuristic uses
Euclidean distance scaled by the minimum positive traversal cost, which keeps
the search consistent with the weighted grid.

Third, the resulting grid route is cached. A route is reused for up to `18`
simulation steps if the same UGV is still assigned to the same target and the
goal cell has not changed. This avoids replanning every frame while still
allowing the controller to react to changed assignments or episode resets.

Finally, the planner exposes only a near-term waypoint rather than the whole
path. The implementation looks ahead up to `10` route cells, stopping early if
the segment to a farther cell would cross a blocked corner or hidden bend. The
waypoint is converted back to world coordinates and passed to the UGV arrival
controller.

## 5. Continuous Ground Control

The route planner chooses where the UGV should go next; it does not directly
control the robot. The low-level controller computes a dimensionless continuous
action from the current position, current velocity, and waypoint direction.
Near the waypoint, the action is reduced by an arrival slowdown term and a
velocity damping term. This keeps UGVs from oscillating around targets or route
corners.

For UGV position $p_t$, waypoint $g_t$, velocity $v_t$, maximum speed
$v_{\max}$, slowdown distance $d_{\mathrm{slow}}$, and damping coefficient
$\eta$, the controller first computes

$$
\hat{d}_t
=
\frac{g_t-p_t}{\left\lVert g_t-p_t\right\rVert_2+\epsilon},
\qquad
\alpha_t
=
\operatorname{clip}
\left(
\frac{\left\lVert g_t-p_t\right\rVert_2}{d_{\mathrm{slow}}},
0,1
\right).
$$

The nominal continuous action is

$$
a_t
=
\operatorname{clip}
\left(
\alpha_t \hat{d}_t
-
\eta \frac{v_t}{v_{\max}},
-1,1
\right).
$$

For UAV lawnmower and ant-colony movement, the same structure is used with the
UAV maximum speed and UAV-specific slowdown distance.

If the direct action toward the waypoint is blocked, the controller tries a
small set of rotated recovery directions, including shallow left/right turns,
right-angle moves, and a reverse direction. It selects the traversable candidate
that makes the most progress toward the waypoint. If no safe candidate exists,
the controller falls back to the previous baseline action or holds position,
depending on the policy.

This creates a two-level UGV behavior:

- A* provides a terrain-aware route at grid scale.
- The arrival controller produces smooth continuous actions at simulator scale.

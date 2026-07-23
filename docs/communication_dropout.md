# Communication Dropout

This document explains how OmniSearch models communication loss and how that
loss affects HAPPO, the heuristic baselines, observations, rewards, and mission
success. The implementation is intentionally abstract: it does not simulate a
radio stack, bandwidth, packets, or a multi-hop network. Instead, it models
whether an agent can participate in the team's shared information state at a
given simulation step.

The core idea is simple:

> Communication dropout changes what each agent knows and what reward/event
> information can be shared. It does not change the physical world.

A UAV can still fly while disconnected. A UGV can still drive while
disconnected. A survivor can still be physically detected or confirmed. What
changes is whether that event reaches the rest of the team immediately.

## 1. Agent-Level Dropout Model

At every step, each agent has a binary communication state:

$$
m_{i,t}
\in
\{0,1\},
$$

where $m_{i,t}=1$ means agent $i$ is connected at time $t$, and $m_{i,t}=0$
means agent $i$ is disconnected. The implementation stores this state on each
agent as `agent.comms_up`.

The dropout parameter

```text
comms_dropout = p
```

is interpreted as the target fraction of time an agent is unavailable. The
simulator supports two processes.

### IID Dropout

In `iid` mode, each agent independently drops out on each step:

$$
m_{i,t}
\sim
\mathrm{Bernoulli}(1-p).
$$

This creates short, memoryless outages. It is useful for testing sensitivity to
random missed communication updates.

### Bursty Dropout

In `bursty` mode, outages persist for several consecutive steps. A connected
agent can start an outage with probability $q$, and the outage duration is
sampled uniformly:

$$
L_i
\sim
\mathrm{Uniform}
\{L_{\min},\ldots,L_{\max}\}.
$$

The current defaults are:

| Parameter | Default |
|---|---:|
| `comms_dropout_mode` | `bursty` |
| `comms_dropout_min_steps` | 5 |
| `comms_dropout_max_steps` | 15 |
| `comms_map_mode` | `per_agent` |

For bursty dropout, the code chooses the outage-start probability so that the
long-run expected down fraction matches `comms_dropout`. With mean outage
duration $\bar{L}$, the start probability is

$$
q
=
\frac{p}{p + (1-p)\bar{L}},
\qquad
\bar{L}
=
\frac{L_{\min}+L_{\max}}{2}.
$$

This keeps the user-facing parameter $p$ comparable between IID and bursty
experiments.

## 2. What Dropout Does and Does Not Mean

Dropout is agent-level, not pairwise link-level. The model does not ask whether
UAV 1 can talk to UGV 2 while UAV 2 can talk to UGV 3. Instead, all agents with
$m_{i,t}=1$ participate in a shared connected team memory at that step, while
agents with $m_{i,t}=0$ are isolated.

Let

$$
\mathcal{C}_t
=
\{i : m_{i,t}=1\}
$$

be the connected set. Agents in $\mathcal{C}_t$ can merge information with one
another. Agents outside $\mathcal{C}_t$ keep acting, but they do not upload
their new information and do not receive the current team memory.

This abstraction is deliberately coarse. It captures the learning problem that
matters here: agents must act from incomplete and sometimes stale mission
knowledge.

## 3. Per-Agent Mission Memory

OmniSearch maintains local mission-memory tensors:

```text
known_survivors_by_agent
confirmed_survivors_by_agent
known_decoys_by_agent
dismissed_decoys_by_agent
```

Direct observations are always written to the observing agent's own memory. For
example, a UAV detection makes that UAV locally know the survivor, and a UGV
confirmation makes that UGV locally confirm the survivor:

$$
K_{i,t+1}(s)
=
K_{i,t}(s)
\lor
E_{i,t}(s),
$$

where $K_{i,t}(s)$ is agent $i$'s local knowledge of survivor $s$, and
$E_{i,t}(s)$ is a direct detection or confirmation event.

Communication controls the merge step. For a connected receiver $i$, survivor
knowledge is synchronized from connected agents:

$$
K_{i,t+1}(s)
=
\bigvee_{j \in \mathcal{C}_t} K_{j,t+1}(s),
\qquad
i \in \mathcal{C}_t.
$$

For a disconnected receiver, the local memory is preserved:

$$
K_{i,t+1}(s)
=
K_{i,t}(s),
\qquad
i \notin \mathcal{C}_t
\quad
\text{unless } i \text{ directly observed } s.
$$

The same pattern is used for confirmations and decoy dismissals.

## 4. Coverage and Confidence Maps

Communication also affects the dense UAV map memories used by observations and
confidence-based reward terms. Runtime behavior always uses per-agent map
buffers:

```text
comm_agent_coverage_grid
comm_agent_confidence_grid
```

There are also connected-team buffers:

```text
comm_team_coverage_grid
comm_team_confidence_grid
```

Each agent updates its own map from its own sensor footprint. Then connected
agents synchronize through the team map. For binary coverage:

$$
M^{\mathrm{cov}}_{\mathrm{team},t+1}
=
M^{\mathrm{cov}}_{\mathrm{team},t}
\lor
\bigvee_{j \in \mathcal{C}_t}
M^{\mathrm{cov}}_{j,t+1}.
$$

Each connected agent receives this merged map:

$$
M^{\mathrm{cov}}_{i,t+1}
=
M^{\mathrm{cov}}_{\mathrm{team},t+1},
\qquad
i \in \mathcal{C}_t.
$$

Disconnected agents keep their private maps. Confidence maps use the same
structure, but with a maximum instead of a logical OR:

$$
M^{\mathrm{conf}}_{\mathrm{team},t+1}(c)
=
\max
\left(
M^{\mathrm{conf}}_{\mathrm{team},t}(c),
\max_{j \in \mathcal{C}_t}
M^{\mathrm{conf}}_{j,t+1}(c)
\right).
$$

This matters because a disconnected UAV may believe an area is still uncertain
even though another UAV has already inspected it, and a connected UAV may not
learn about a disconnected UAV's new confidence gain until communication
returns.

## 5. Observation Effects

Dropout affects observations through the information state, not by adding a
large explicit "communication feature vector" everywhere.

For survivor-message observations, the queried agent receives only what is in
its local memory plus information from currently connected agents. If the agent
is disconnected, its observation is based on its stale local memory. In
conceptual terms:

$$
\tilde{K}_{i,t}(s)
=
\begin{cases}
\bigvee_{j \in \mathcal{C}_t} K_{j,t}(s), & i \in \mathcal{C}_t,\\
K_{i,t}(s), & i \notin \mathcal{C}_t.
\end{cases}
$$

For UAV map observations, the same principle applies to coverage and confidence
grids. A connected UAV observes the synchronized team map; a disconnected UAV
observes its private map.

This means dropout can create three practical failure modes:

- duplicate search, because an agent does not know that a teammate already
  covered an area;
- delayed handoff, because UGVs do not immediately receive new survivor
  detections;
- stale assignment state, because disconnected UGVs cannot publish target
  changes.

## 6. Reward Effects

Sparse team rewards are communication-gated. This prevents a disconnected event
from instantly rewarding the entire team, which would leak information through
the reward channel.

Let $e_t(s)$ denote a newly scouted or newly confirmed survivor event, and let
$A_t(s)$ be the set of agents that directly caused that event. A team event
reward is broadcast only when at least one event actor is connected:

$$
B_t(s)
=
\mathbb{1}
\left[
\exists j \in A_t(s)
\;:\;
m_{j,t}=1
\right].
$$

Then agent $i$ receives the team event reward only if it is also connected:

$$
r^{\mathrm{team}}_{i,t}(s)
=
w_{\mathrm{event}}\,
e_t(s)\,
B_t(s)\,
m_{i,t}.
$$

A disconnected direct observer still records its own local event, but it does
not broadcast the sparse team reward on that step. The reward is intentionally
not replayed later after reconnection. This makes the reward signal match the
communication-limited information flow.

## 7. Mission Success and Communicated Confirmation

Physical confirmation and communicated confirmation are distinct.

When a UGV reaches a survivor, the simulator can mark that survivor as
physically confirmed. However, under dropout, mission-level completion waits
until the confirmation has reached the connected team network. The simulator
therefore maintains

```text
communicated_confirmed_survivors
```

and updates it only from connected agents:

$$
C^{\mathrm{comm}}_{t+1}(s)
=
C^{\mathrm{comm}}_t(s)
\lor
\bigvee_{i \in \mathcal{C}_t}
C_{i,t+1}(s).
$$

This is why communication dropout can affect both confirmation timing and
success rate even when the UGV physically reaches the survivor. The mission is
not fully resolved until the team has received the result.

## 8. HAPPO Under Dropout

HAPPO is affected by dropout through the environment interface. The actor still
maps its observation to a continuous action, and the centralized critic is still
used during training. What changes is the content of each agent's observation
and the reward information available at that step.

For HAPPO, dropout affects:

- survivor and decoy message observations;
- per-agent coverage and confidence maps;
- UGV assignment and planner-hint observations, because targetability is based
  on local UGV knowledge;
- team scout and confirm rewards;
- communicated mission-completion timing.

The learned policy can therefore adapt to missing information only if it has
seen similar communication conditions during training or if its behavior is
robust enough to generalize. A policy trained with perfect communication may
depend strongly on synchronized maps and immediate handoffs. A policy evaluated
with dropout may still search effectively, but it can lose time through delayed
UGV dispatch or redundant UAV coverage.

## 9. Baseline-Specific Effects

### Random Action

`random_action` samples UAV actions independently of communication. Dropout
does not change the UAV search motion itself. It changes when UGVs learn about
survivor detections and which targets they are allowed to pursue.

If a UGV is disconnected, it keeps its current target lease if it already had
one. If it has no valid target, it cannot accept newly shared work until
communication returns.

### Persistent Random Walk

`random_walk` is similar to `random_action`, but each agent keeps a persistent
heading. Communication does not change the random-walk dynamics. It changes the
switch from uninformed movement to target-directed UGV routing.

Under dropout, a UGV may continue its previous assignment while isolated, or it
may keep walking without receiving a newly scouted survivor.

### Lawnmower

`lawnmower` UAVs follow precomputed land-aware sweep lanes, so dropout does not
change their coverage path. The main effect is on the UGV side: ground robots
only pursue locally known pending survivors.

This makes `lawnmower` a useful baseline for separating search coverage from
handoff robustness. The aerial coverage can remain strong while confirmation
degrades because survivor detections arrive late to UGVs.

### Highest Confidence

`highest_confidence` uses lawnmower UAV coverage, but UGV target priority is
based on the strongest retained detection confidence visible through the
connected team. During dropout, disconnected agents do not contribute new
confidence values to the connected priority pool and do not receive priority
updates from others.

Conceptually, the priority for survivor $s$ is

$$
P_t(s)
=
\max_{i \in \mathcal{C}_t}
H_{i,t}(s),
$$

where $H_{i,t}(s)$ is agent $i$'s retained detection confidence for survivor
$s$. A disconnected agent keeps its private confidence memory but does not
update the team's priority ordering until it reconnects.

### Ant-Colony

`ant_colony` is the baseline most directly shaped by dropout. Each UAV has a
private recency map, and connected UAVs merge their timestamp maps by taking
the newest observation time per cell:

$$
T_{i,t+1}(c)
=
\max_{j \in \mathcal{C}_t} T_{j,t+1}(c),
\qquad i \in \mathcal{C}_t.
$$

Disconnected UAVs keep their private maps. This can be helpful or harmful:
private maps preserve autonomous exploration, but disconnected agents may
revisit areas already covered by teammates. When communication returns, the
recency maps merge and the search pattern becomes coordinated again.

UGVs in `ant_colony` also use local event memory. A disconnected UGV can
continue toward a leased target, but newly discovered targets from other agents
do not become available until reconnection.

## 10. UGV Assignment Leases

The lease mechanism is important for both HAPPO planner-enabled runs and
communication-aware baselines. Suppose UGV $g$ was assigned target $s$ before
dropout. If $g$ becomes disconnected, the assignment is frozen:

$$
\ell_{g,t+1}
=
\ell_{g,t}
\qquad
\text{if } m_{g,t}=0.
$$

Connected UGVs solve their assignment problem over the remaining unreserved
targets:

$$
S^{\mathrm{available}}_t
=
S^{\mathrm{pending}}_t
\setminus
\{\ell_{g,t}: m_{g,t}=0\}.
$$

Only connected UGVs can publish assignment changes. This avoids unrealistic
instantaneous team reassignment when a robot drops out, while still allowing the
isolated robot to keep moving toward its last known target.

## 11. Diagnostics and CLI Options

Communication settings can be controlled in the main diagnostic and trajectory
scripts:

```bash
--comms-dropout 0.3
--comms-dropout-mode iid
--comms-dropout-mode bursty
--comms-dropout-min-steps 5
--comms-dropout-max-steps 15
--comms-map-mode per_agent
```

For joint baseline diagnostics, the scenario defaults are set to communication
aware per-agent maps:

```text
comms_map_mode = per_agent
comms_dropout_mode = bursty
```

The web trajectory export also writes `comms_up` for each agent, so the viewer
can display which agents are connected or disconnected at each frame.

## 12. Interpretation

Communication dropout mainly tests robustness of coordination, not raw
mobility. A policy or heuristic may still cover terrain well while failing to
confirm survivors quickly, because confirmation requires handoff from aerial
detection to ground routing. The strongest approaches should therefore preserve
three properties under dropout:

- UAVs avoid excessive duplicate coverage despite stale map memories.
- UGVs continue useful assignments during temporary isolation.
- Confirmations are communicated early enough to count toward mission success.

This is why dropout experiments are most informative when evaluated with
confirmation AUC, time to confirmation, and success rate, not only final
coverage or confidence.

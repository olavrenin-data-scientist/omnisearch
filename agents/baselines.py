"""
Hand-coded baseline coordination strategies.

These are the comparison baselines for the capstone's central claim that
HAPPO/MAPPO coordination beats simpler heuristics. Each strategy is a
callable ``policy(env) -> per-agent actions`` and plugs into the same
evaluation harness as a trained policy (see
``evaluation.mission_metrics.evaluate_policy``).

The baselines in this file mirror the strategies named in the project
plan:

    Strategy              How drones decide        How ground robots decide
    --------------------  -----------------------  -------------------------
    RandomPolicy          random within range      random within range
    LawnmowerPolicy       sweep a serpentine path  follow nearest scouted
    NearestCandidate      random walk              go to nearest survivor
    HighestConfidence     bias toward unscouted    go to most-recently scouted

The "candidate" abstraction in the plan (uncertain detections with
confidence scores) is currently a stretch — for the MVP we use
ground-truth scout/found masks from the scenario directly. When the
probabilistic sensor model lands, swap the lookups for belief-map
queries; the policy interface stays the same.
"""

from __future__ import annotations

import math
from typing import Callable, List

import torch

from agents.pathfinding import find_ground_route
from envs.wildfire_search import WildfireSearchScenario, X, Y


# Each policy returns a list of (B, action_dim) action tensors, one per
# agent, in the same order as env.agents. WildfireSearchScenario's
# actions are 2D continuous in [-1, 1] (a force vector).

# Immediate collision check for the next UGV action. This should stay close to
# one physical step; long-horizon obstacle avoidance is handled by A* routing.
GROUND_LOOKAHEAD = 0.045
GROUND_ROUTE_WAYPOINT_CELLS = 10
GROUND_ROUTE_REPLAN_STEPS = 18
GROUND_ROUTE_FIRE_PENALTY = 25.0
GROUND_RECOVERY_ANGLES = (
    0.0,
    math.pi / 6,
    -math.pi / 6,
    math.pi / 3,
    -math.pi / 3,
    math.pi / 2,
    -math.pi / 2,
    math.pi,
)


def _rotate(actions: torch.Tensor, angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    x = actions[:, X] * c - actions[:, Y] * s
    y = actions[:, X] * s + actions[:, Y] * c
    return torch.stack([x, y], dim=-1)


def _grid_to_world(sc: WildfireSearchScenario, gx: int, gy: int, device: torch.device) -> torch.Tensor:
    cell_w = 2 * sc.x_semidim / sc.fire_grid_size
    cell_h = 2 * sc.y_semidim / sc.fire_grid_size
    return torch.tensor(
        [
            -sc.x_semidim + (gx + 0.5) * cell_w,
            -sc.y_semidim + (gy + 0.5) * cell_h,
        ],
        dtype=torch.float,
        device=device,
    )


def _nearest_traversable_cell(sc: WildfireSearchScenario, env_index: int, gx: int, gy: int) -> tuple[int, int] | None:
    traversable = sc.traversable_grid[env_index]
    size = sc.fire_grid_size
    gx = max(0, min(size - 1, gx))
    gy = max(0, min(size - 1, gy))
    if bool(traversable[gy, gx].item()):
        return gx, gy
    max_radius = min(size, 18)
    for radius in range(1, max_radius + 1):
        y0, y1 = max(0, gy - radius), min(size - 1, gy + radius)
        x0, x1 = max(0, gx - radius), min(size - 1, gx + radius)
        candidates = []
        for y in range(y0, y1 + 1):
            candidates.extend(((x0, y), (x1, y)))
        for x in range(x0 + 1, x1):
            candidates.extend(((x, y0), (x, y1)))
        valid = [(x, y) for x, y in candidates if bool(traversable[y, x].item())]
        if valid:
            return min(valid, key=lambda cell: (cell[0] - gx) ** 2 + (cell[1] - gy) ** 2)
    return None


def _route_ground_waypoints(
    sc: WildfireSearchScenario,
    ground_index: int,
    target_pos: torch.Tensor,
    target_indices: torch.Tensor,
    route_cache: List[dict] | None = None,
) -> torch.Tensor:
    """Return near-term A* waypoints for UGVs whose direct route is blocked."""
    ag_idx = sc.n_drones + ground_index
    pos = sc.world.agents[ag_idx].state.pos
    direct_ok = sc._path_is_traversable(pos.unsqueeze(1), target_pos.unsqueeze(1)).squeeze(1)
    waypoint = target_pos.clone()
    start_gx, start_gy = sc._positions_to_grid(pos.unsqueeze(1))
    goal_gx, goal_gy = sc._positions_to_grid(target_pos.unsqueeze(1))
    for b in range(sc.world.batch_dim):
        if bool(direct_ok[b].item()):
            if route_cache is not None:
                route_cache[ground_index].pop(int(b), None)
            continue
        start = _nearest_traversable_cell(sc, b, int(start_gx[b, 0]), int(start_gy[b, 0]))
        goal = _nearest_traversable_cell(sc, b, int(goal_gx[b, 0]), int(goal_gy[b, 0]))
        if start is None or goal is None:
            continue
        cache_entry = None if route_cache is None else route_cache[ground_index].get(int(b))
        step = int(sc.step_count[b].item())
        target_id = int(target_indices[b].item())
        path = []
        if cache_entry is not None:
            still_fresh = step - cache_entry["step"] < GROUND_ROUTE_REPLAN_STEPS
            same_target = cache_entry["target_id"] == target_id
            same_goal = cache_entry["goal"] == goal
            if still_fresh and same_target and same_goal:
                path = cache_entry["path"]
        if not path:
            route_cost = sc.mobility_cost_grid[b].clone()
            route_cost = route_cost + sc.fire_grid[b].float() * GROUND_ROUTE_FIRE_PENALTY
            path = find_ground_route(
                traversable=sc.traversable_grid[b],
                movement_cost=route_cost,
                start=start,
                goal=goal,
            )
            if route_cache is not None:
                route_cache[ground_index][int(b)] = {
                    "step": step,
                    "target_id": target_id,
                    "goal": goal,
                    "path": path,
                }
        if len(path) < 2:
            continue
        nearest_idx = min(
            range(len(path)),
            key=lambda idx: (path[idx][0] - start[0]) ** 2 + (path[idx][1] - start[1]) ** 2,
        )
        wx, wy = path[min(nearest_idx + GROUND_ROUTE_WAYPOINT_CELLS, len(path) - 1)]
        waypoint[b] = _grid_to_world(sc, wx, wy, target_pos.device)
    return waypoint


def _terrain_safe_ground_action(
    sc: WildfireSearchScenario,
    ground_index: int,
    target_pos: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Choose a traversable action, backing out when direct motion is blocked."""
    ag_idx = sc.n_drones + ground_index
    pos = sc.world.agents[ag_idx].state.pos
    to_target = target_pos - pos
    distance = to_target.norm(dim=-1, keepdim=True)
    direction = to_target / distance.clamp_min(1e-6)
    magnitude = distance.clamp(max=1.0)
    direct = direction * magnitude

    candidates = torch.stack(
        [_rotate(direct, angle) for angle in GROUND_RECOVERY_ANGLES],
        dim=1,
    )
    endpoints = pos.unsqueeze(1) + candidates * GROUND_LOOKAHEAD
    endpoints[..., X] = endpoints[..., X].clamp(-sc.x_semidim, sc.x_semidim)
    endpoints[..., Y] = endpoints[..., Y].clamp(-sc.y_semidim, sc.y_semidim)

    start = pos.unsqueeze(1).expand_as(endpoints)
    traversable = sc._path_is_traversable(start, endpoints)
    new_distance = (target_pos.unsqueeze(1) - endpoints).norm(dim=-1)
    progress = distance.squeeze(-1).unsqueeze(1) - new_distance

    # Prefer routes that still make progress; if all forward/side options are
    # blocked, the 180-degree candidate acts as a controlled return/back-out.
    score = torch.where(
        traversable,
        progress,
        torch.full_like(progress, float("-inf")),
    )
    best = score.argmax(dim=-1)
    chosen = candidates.gather(1, best.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    any_safe = traversable.any(dim=-1)
    return torch.where(any_safe.unsqueeze(-1), chosen, fallback)


def _coordinated_ground_actions(
    sc: WildfireSearchScenario,
    targetable: torch.Tensor,
    fallback_actions: List[torch.Tensor],
    priority: torch.Tensor | None = None,
    route_cache: List[dict] | None = None,
) -> List[torch.Tensor]:
    """Assign at most one ground robot to each targetable survivor.

    With ``priority=None`` each UGV greedily picks its nearest unassigned
    survivor. With priority scores, each UGV picks the highest-priority
    unassigned survivor. Extra UGVs use their fallback action.
    """
    if sc.n_ground == 0:
        return []

    B = sc.world.batch_dim
    surv_pos = torch.stack([s.state.pos for s in sc._survivors], dim=1)
    assigned = torch.zeros_like(targetable)
    actions = [a.clone() for a in fallback_actions]
    batch_idx = torch.arange(B, device=targetable.device)

    for gi in range(sc.n_ground):
        ag_idx = sc.n_drones + gi
        pos = sc.world.agents[ag_idx].state.pos
        available = targetable & ~assigned
        any_targetable = available.any(dim=-1)

        if priority is None:
            d = (surv_pos - pos.unsqueeze(1)).norm(dim=-1)
            scores = -d
        else:
            scores = priority

        masked_scores = torch.where(
            available, scores, torch.full_like(scores, float("-inf")),
        )
        best = masked_scores.argmax(dim=-1)
        target_pos = surv_pos.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
        waypoint = _route_ground_waypoints(sc, gi, target_pos, best, route_cache)
        delta = _terrain_safe_ground_action(sc, gi, waypoint, actions[gi])
        actions[gi] = torch.where(any_targetable.unsqueeze(-1), delta, actions[gi])

        if any_targetable.any():
            assigned[batch_idx[any_targetable], best[any_targetable]] = True

    return actions


# ----------------------------------------------------------------------
# Random
# ----------------------------------------------------------------------
class RandomPolicy:
    """All agents take random in-range actions. The reference 'do nothing
    smart' baseline."""

    def __call__(self, env) -> List[torch.Tensor]:
        return env.get_random_actions()


# ----------------------------------------------------------------------
# Lawnmower (drones) + nearest-confirm (ground)
# ----------------------------------------------------------------------
class LawnmowerPolicy:
    """
    Drones execute a deterministic horizontal serpentine sweep that
    eventually covers the whole map.

    Ground robots head to the nearest *scouted* survivor (using the
    scenario's `scouted_survivors` mask). If no survivor has been
    scouted yet they hold position.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        self.t = 0
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]
        # Each drone gets a y-band — distribute them vertically
        self.drone_band_y = self._drone_band_ys(self.scenario.n_drones)

    @staticmethod
    def _drone_band_ys(n: int) -> List[float]:
        if n == 1:
            return [0.0]
        return [(-0.7 + 1.4 * i / (n - 1)) for i in range(n)]

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B  = sc.world.batch_dim
        device = sc.fire_grid.device
        out: List[torch.Tensor] = []

        # ---- Drones: serpentine x-sweep, y-band fixed per drone ----
        period = 80  # steps for one half-cycle
        phase = (self.t // period) % 2
        x_target = +0.9 if phase == 0 else -0.9
        for i in range(sc.n_drones):
            pos = sc.world.agents[i].state.pos  # (B, 2)
            dx = (x_target - pos[:, X]).clamp(-1.0, 1.0)
            dy = (self.drone_band_y[i] - pos[:, Y]).clamp(-1.0, 1.0)
            out.append(torch.stack([dx, dy], dim=-1))

        # ---- Ground robots: split up across scouted survivors ----
        scouted = sc.scouted_survivors        # (B, S) bool
        found   = sc.found_survivors          # (B, S) bool
        targetable = scouted & ~found         # not yet confirmed

        hold_actions = [
            torch.zeros(B, 2, device=device)
            for _ in range(sc.n_ground)
        ]
        out.extend(_coordinated_ground_actions(
            sc, targetable, hold_actions, route_cache=self.ground_route_cache,
        ))

        self.t += 1
        return out


# ----------------------------------------------------------------------
# Nearest-candidate (drones random walk, ground -> nearest scouted)
# ----------------------------------------------------------------------
class NearestCandidatePolicy:
    """
    Drones take random actions (no map coverage strategy).
    Ground robots head to the *nearest* survivor that has been scouted
    by any drone but not yet confirmed.

    This is the canonical "obvious heuristic" baseline that HAPPO should
    beat — it's reactive, doesn't reason about staleness or hazard.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        out: List[torch.Tensor] = []

        # Drones: random
        rand_actions = env.get_random_actions()
        for i in range(sc.n_drones):
            out.append(rand_actions[i])

        # Ground: split up across nearest scouted survivors
        scouted    = sc.scouted_survivors
        found      = sc.found_survivors
        targetable = scouted & ~found
        fallback = [
            rand_actions[sc.n_drones + gi]
            for gi in range(sc.n_ground)
        ]
        out.extend(_coordinated_ground_actions(
            sc, targetable, fallback, route_cache=self.ground_route_cache,
        ))

        return out


# ----------------------------------------------------------------------
# Highest-confidence-first (proxy: most-recently-scouted = freshest)
# ----------------------------------------------------------------------
class HighestConfidencePolicy:
    """
    Drones cover the area (lawnmower).
    Ground robots prioritize the *most recently* scouted survivor —
    proxy for the highest-confidence candidate in the candidate-belief
    abstraction (the plan's intended formulation). Will be swapped for
    a real confidence lookup once the probabilistic sensor model is in.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        self.scout_step = torch.full(
            (self.scenario.world.batch_dim, self.scenario.n_survivors),
            -1,
            dtype=torch.float,
            device=self.scenario.fire_grid.device,
        )
        self.t = 0
        self._lawnmower = LawnmowerPolicy(env)
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        newly_scouted = sc.scouted_survivors & (self.scout_step < 0)
        self.scout_step = torch.where(
            newly_scouted,
            torch.full_like(self.scout_step, float(self.t)),
            self.scout_step,
        )

        actions = self._lawnmower(env)

        # Replace ground actions to use freshest unassigned scouted targets.
        found = sc.found_survivors
        targetable = (self.scout_step >= 0) & ~found
        fallback = [
            actions[sc.n_drones + gi]
            for gi in range(sc.n_ground)
        ]
        coordinated = _coordinated_ground_actions(
            sc,
            targetable,
            fallback,
            priority=self.scout_step,
            route_cache=self.ground_route_cache,
        )
        for gi, action in enumerate(coordinated):
            actions[sc.n_drones + gi] = action

        self.t += 1
        return actions


# ----------------------------------------------------------------------
# Registry — for use from scripts/notebooks
# ----------------------------------------------------------------------
BASELINES: dict[str, Callable] = {
    "random":              RandomPolicy,
    "lawnmower":           LawnmowerPolicy,
    "nearest_candidate":   NearestCandidatePolicy,
    "highest_confidence":  HighestConfidencePolicy,
}


def get_baseline(name: str, env) -> Callable:
    """Resolve a baseline by name, constructing it on the given env."""
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name!r}. Available: {list(BASELINES)}")
    cls = BASELINES[name]
    # RandomPolicy doesn't need env, others do
    return cls(env) if cls is not RandomPolicy else cls()

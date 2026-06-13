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
    RandomActionPolicy    random within range      random within range
    RandomWalkPolicy      persistent random walk   persistent random walk
    LawnmowerPolicy       sweep a serpentine path  follow nearest scouted
    NearestCandidate      random walk              go to nearest survivor
    HighestConfidence     bias toward unscouted    go to most-recently scouted
    AntColony             avoid fresh pheromone    follow locally known survivors

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
from envs.wildfire_search import LAND_ROCK, LAND_WATER, WildfireSearchScenario, X, Y


# Each policy returns a list of (B, action_dim) action tensors, one per
# agent, in the same order as env.agents. WildfireSearchScenario's
# actions are 2D continuous in [-1, 1] (a force vector).

GROUND_ROUTE_REPLAN_STEPS = 18
GROUND_ROUTE_FIRE_PENALTY = 25.0
GROUND_ROUTE_MAX_LOOKAHEAD_CELLS = 10
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
# _search_footprint() returns camera radius, so 1.2 * radius = 0.6 * full footprint width.
DRONE_LANE_SPACING_FACTOR = 1.2
DRONE_WAYPOINT_TOLERANCE_M = 5.0
DRONE_ARRIVAL_SLOWDOWN_M = 40.0
DRONE_ARRIVAL_DAMPING = 0.65
DRONE_CRUISE_ACTION = 0.95
PHEROMONE_UNSEEN_STEP = -1
RANDOM_WALK_PERSISTENCE_S = 20.0
RANDOM_WALK_ACTION = 0.95


def _rotate(actions: torch.Tensor, angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    x = actions[:, X] * c - actions[:, Y] * s
    y = actions[:, X] * s + actions[:, Y] * c
    return torch.stack([x, y], dim=-1)


def _reflect_random_walk_directions(
    positions: torch.Tensor,
    directions: torch.Tensor,
    step_distance: torch.Tensor,
    x_bound: float,
    y_bound: float,
) -> torch.Tensor:
    """Reflect headings whose next nominal step would cross the world edge."""
    proposed = positions + directions * step_distance
    reflected = directions.clone()
    hit_x = (proposed[..., X] < -x_bound) | (proposed[..., X] > x_bound)
    hit_y = (proposed[..., Y] < -y_bound) | (proposed[..., Y] > y_bound)
    reflected[..., X] = torch.where(hit_x, -reflected[..., X], reflected[..., X])
    reflected[..., Y] = torch.where(hit_y, -reflected[..., Y], reflected[..., Y])
    return reflected


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


def _grid_segment_is_traversable(
    traversable: torch.Tensor,
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    """Check a cell-center segment without skipping blocked corner cells."""
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return bool(traversable[y0, x0].item())

    previous = start
    for i in range(steps + 1):
        x = int(round(x0 + (x1 - x0) * i / steps))
        y = int(round(y0 + (y1 - y0) * i / steps))
        if not bool(traversable[y, x].item()):
            return False
        px, py = previous
        if x != px and y != py:
            if not bool(traversable[py, x].item()):
                return False
            if not bool(traversable[y, px].item()):
                return False
        previous = (x, y)
    return True


def _route_lookahead_cell(
    traversable: torch.Tensor,
    path: list[tuple[int, int]],
    nearest_idx: int,
) -> tuple[int, int]:
    """Return the farthest near-term route cell visible from the current cell."""
    start = path[nearest_idx]
    best = path[min(nearest_idx + 1, len(path) - 1)]
    stop = min(nearest_idx + GROUND_ROUTE_MAX_LOOKAHEAD_CELLS, len(path) - 1)
    for idx in range(nearest_idx + 2, stop + 1):
        candidate = path[idx]
        if not _grid_segment_is_traversable(traversable, start, candidate):
            break
        best = candidate
    return best


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
            step_went_backward = step < cache_entry["step"]
            still_fresh = (step - cache_entry["step"]) < GROUND_ROUTE_REPLAN_STEPS
            same_target = cache_entry["target_id"] == target_id
            same_goal = cache_entry["goal"] == goal
            if not step_went_backward and still_fresh and same_target and same_goal:
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
        # Keep enough lookahead for smooth motion, but stop before the route
        # bends behind an obstacle or across a blocked diagonal corner.
        wx, wy = _route_lookahead_cell(sc.traversable_grid[b], path, nearest_idx)
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
    vel = sc.world.agents[ag_idx].state.vel
    vmax = max(float(getattr(sc, "ground_max_speed_sim", 0.0)), 1e-6)
    slowdown_distance = (
        sc.ground_arrival_slowdown_m
        * sc.terrain_sim_units_per_meter.to(distance.device)
    ).view(-1, 1).clamp_min(torch.finfo(distance.dtype).eps)
    direct = _ground_arrival_action(
        direction=direction,
        distance=distance,
        velocity=vel,
        max_speed=vmax,
        slowdown_distance=slowdown_distance,
        damping=sc.ground_arrival_damping,
    )

    candidates = torch.stack(
        [_rotate(direct, angle) for angle in GROUND_RECOVERY_ANGLES],
        dim=1,
    )
    lookahead = max(
        sc.ground_speed_mps * sc.sim_step_seconds * float(sc.terrain_sim_units_per_meter.max()),
        1e-6,
    )
    endpoints = pos.unsqueeze(1) + candidates * lookahead
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
    return torch.where(any_safe.unsqueeze(-1), chosen, fallback).clamp(-1.0, 1.0)


def _ground_arrival_action(
    *,
    direction: torch.Tensor,
    distance: torch.Tensor,
    velocity: torch.Tensor,
    max_speed: float,
    slowdown_distance: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Return a dimensionless, scale-independent PD arrival action."""
    proportional = (distance / slowdown_distance).clamp(0.0, 1.0)
    brake = (velocity / max(float(max_speed), 1e-6)).clamp(-1.0, 1.0)
    return (direction * proportional - float(damping) * brake).clamp(-1.0, 1.0)


def _drone_arrival_action(
    *,
    direction: torch.Tensor,
    distance: torch.Tensor,
    velocity: torch.Tensor,
    max_speed: float,
    slowdown_distance: float,
) -> torch.Tensor:
    """Brake into lawnmower endpoints so the next lane starts aligned."""
    proportional = min(max(distance / max(float(slowdown_distance), 1e-6), 0.0), 1.0)
    desired = direction * (DRONE_CRUISE_ACTION * proportional)
    brake = velocity / max(float(max_speed), 1e-6)
    return (desired - DRONE_ARRIVAL_DAMPING * brake).clamp(-1.0, 1.0)


def _merge_local_timestamp_maps(
    local_maps: torch.Tensor,
    comms_up: torch.Tensor,
) -> torch.Tensor:
    """Merge team timestamps into agents that can currently receive comms."""
    team_latest = local_maps.amax(dim=1, keepdim=True)
    return torch.where(comms_up[:, :, None, None], team_latest, local_maps)


def _merge_local_bool_knowledge(
    local_knowledge: torch.Tensor,
    comms_up: torch.Tensor,
) -> torch.Tensor:
    """Merge monotonic event knowledge into agents with a live receiver."""
    team_knowledge = local_knowledge.any(dim=1, keepdim=True)
    return torch.where(comms_up[:, :, None], team_knowledge, local_knowledge)


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
# Random action
# ----------------------------------------------------------------------
class RandomActionPolicy:
    """All agents take random in-range actions. The reference 'do nothing
    smart' baseline."""

    def __call__(self, env) -> List[torch.Tensor]:
        return env.get_random_actions()


# ----------------------------------------------------------------------
# Persistent random walk
# ----------------------------------------------------------------------
class RandomWalkPolicy:
    """Uninformed correlated search with boundary and terrain reactions.

    Each agent retains a heading whose angle follows rotational diffusion.
    The policy does not inspect survivors, fire, coverage, or communication.
    """

    def __init__(self, env, persistence_s: float = RANDOM_WALK_PERSISTENCE_S):
        if persistence_s <= 0.0:
            raise ValueError("persistence_s must be positive")
        self.scenario: WildfireSearchScenario = env.scenario
        self.persistence_s = float(persistence_s)
        sc = self.scenario
        self.headings = torch.empty(
            sc.world.batch_dim,
            len(sc.world.agents),
            device=sc.fire_grid.device,
        )
        self.previous_step_count = sc.step_count.clone()
        self._randomize_environments(torch.ones_like(sc.step_count, dtype=torch.bool))

    def __call__(self, env) -> List[torch.Tensor]:
        del env
        sc = self.scenario
        reset = sc.step_count < self.previous_step_count
        self._randomize_environments(reset)

        angular_std = math.sqrt(2.0 * float(sc.sim_step_seconds) / self.persistence_s)
        self.headings.add_(angular_std * torch.randn_like(self.headings))
        directions = torch.stack(
            (self.headings.cos(), self.headings.sin()),
            dim=-1,
        )

        positions = torch.stack(
            [agent.state.pos for agent in sc.world.agents],
            dim=1,
        )
        sim_per_meter = sc.terrain_sim_units_per_meter.view(-1, 1, 1)
        distances_m = torch.tensor(
            [
                sc.drone_speed_mps * sc.sim_step_seconds
                if idx < sc.n_drones
                else sc.ground_speed_mps * sc.sim_step_seconds
                for idx in range(len(sc.world.agents))
            ],
            dtype=positions.dtype,
            device=positions.device,
        ).view(1, -1, 1)
        step_distance = sim_per_meter * distances_m
        x_bound = max(float(sc.x_semidim) - float(sc.agent_radius), 0.0)
        y_bound = max(float(sc.y_semidim) - float(sc.agent_radius), 0.0)
        directions = _reflect_random_walk_directions(
            positions,
            directions,
            step_distance,
            x_bound,
            y_bound,
        )
        self.headings.copy_(torch.atan2(directions[..., Y], directions[..., X]))

        actions = [
            directions[:, idx] * RANDOM_WALK_ACTION
            for idx in range(sc.n_drones)
        ]
        actions.extend(self._ground_actions(directions, step_distance))
        self.previous_step_count = sc.step_count.clone()
        return actions

    def _randomize_environments(self, mask: torch.Tensor) -> None:
        count = int(mask.sum().item())
        if count:
            self.headings[mask] = 2.0 * math.pi * torch.rand(
                count,
                self.headings.shape[1],
                device=self.headings.device,
            )

    def _ground_actions(
        self,
        directions: torch.Tensor,
        step_distance: torch.Tensor,
    ) -> List[torch.Tensor]:
        sc = self.scenario
        actions: List[torch.Tensor] = []
        for ground_idx in range(sc.n_ground):
            agent_idx = sc.n_drones + ground_idx
            position = sc.world.agents[agent_idx].state.pos
            base = directions[:, agent_idx]
            candidates = torch.stack(
                [_rotate(base, angle) for angle in GROUND_RECOVERY_ANGLES],
                dim=1,
            )
            endpoints = position.unsqueeze(1) + (
                candidates * step_distance[:, agent_idx].unsqueeze(1)
            )
            in_bounds = (
                (endpoints[..., X] >= -sc.x_semidim + sc.agent_radius)
                & (endpoints[..., X] <= sc.x_semidim - sc.agent_radius)
                & (endpoints[..., Y] >= -sc.y_semidim + sc.agent_radius)
                & (endpoints[..., Y] <= sc.y_semidim - sc.agent_radius)
            )
            starts = position.unsqueeze(1).expand_as(endpoints)
            safe = in_bounds & sc._path_is_traversable(starts, endpoints)
            choice = safe.to(torch.int64).argmax(dim=1)
            selected = candidates.gather(
                1, choice.view(-1, 1, 1).expand(-1, 1, 2),
            ).squeeze(1)
            any_safe = safe.any(dim=1)
            action = torch.where(
                any_safe.unsqueeze(-1),
                selected * RANDOM_WALK_ACTION,
                torch.zeros_like(selected),
            )
            reverse = torch.atan2(-base[:, Y], -base[:, X])
            selected_heading = torch.atan2(selected[:, Y], selected[:, X])
            self.headings[:, agent_idx] = torch.where(
                any_safe, selected_heading, reverse,
            )
            actions.append(action)
        return actions


# ----------------------------------------------------------------------
# Lawnmower (drones) + nearest-confirm (ground)
# ----------------------------------------------------------------------
class LawnmowerPolicy:
    """
    Drones execute workload-balanced boustrophedon coverage over searchable
    terrain. Water/ocean and rock cells do not contribute search workload,
    and each lane is trimmed to the land segment it covers.

    Ground robots head to the nearest *scouted* survivor (using the
    scenario's `scouted_survivors` mask). If no survivor has been
    scouted yet they hold position.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        self.t = 0
        self._prev_step_max = -1
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]
        self.drone_waypoints: list[list[list[tuple[float, float]]]] | None = None
        self.drone_waypoint_index: torch.Tensor | None = None
        self.drone_plan_signature: tuple | None = None

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B  = sc.world.batch_dim
        device = sc.fire_grid.device
        out: List[torch.Tensor] = []

        # ---- Drones: land-aware, workload-balanced lawnmower coverage ----
        out.extend(self._drone_lawnmower_actions())

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

    def _drone_lawnmower_actions(self) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        device = sc.fire_grid.device
        self._ensure_drone_lawnmower_plans()
        if sc.n_drones == 0:
            return []

        assert self.drone_waypoints is not None
        assert self.drone_waypoint_index is not None

        actions: List[torch.Tensor] = []
        for drone_idx in range(sc.n_drones):
            pos = sc.world.agents[drone_idx].state.pos
            action = torch.zeros(B, 2, device=device)
            for b in range(B):
                waypoints = self.drone_waypoints[b][drone_idx]
                if not waypoints:
                    continue
                wp_idx = int(self.drone_waypoint_index[b, drone_idx].item())
                wp_idx = min(wp_idx, len(waypoints) - 1)
                target = torch.tensor(waypoints[wp_idx], dtype=torch.float, device=device)
                delta = target - pos[b]
                distance = float(delta.norm().item())
                waypoint_tolerance = self._waypoint_tolerance(b)
                prev_wp = torch.tensor(waypoints[(wp_idx - 1) % len(waypoints)], dtype=torch.float, device=device)
                overshot = float(torch.dot(delta, target - prev_wp).item()) < 0.0
                if distance <= waypoint_tolerance or overshot:
                    wp_idx = (wp_idx + 1) % len(waypoints)
                    self.drone_waypoint_index[b, drone_idx] = wp_idx
                    target = torch.tensor(waypoints[wp_idx], dtype=torch.float, device=device)
                    delta = target - pos[b]
                    distance = float(delta.norm().item())
                direction = delta / delta.norm().clamp_min(1e-6)
                slowdown_distance = max(
                    DRONE_ARRIVAL_SLOWDOWN_M
                    * float(sc.terrain_sim_units_per_meter[b].item()),
                    waypoint_tolerance,
                )
                action[b] = _drone_arrival_action(
                    direction=direction,
                    distance=distance,
                    velocity=sc.world.agents[drone_idx].state.vel[b],
                    max_speed=float(sc.drone_max_speed_sim),
                    slowdown_distance=slowdown_distance,
                )
            actions.append(action)
        return actions

    def _ensure_drone_lawnmower_plans(self) -> None:
        sc = self.scenario
        B = sc.world.batch_dim
        if sc.n_drones == 0:
            return
        footprint = self._search_footprint()
        lane_spacing = footprint * DRONE_LANE_SPACING_FACTOR
        land_counts = tuple(
            int(((sc.land_cover_grid[b] != LAND_WATER) & (sc.land_cover_grid[b] != LAND_ROCK)).sum().item())
            for b in range(B)
        )
        signature = (
            B,
            sc.n_drones,
            sc.fire_grid_size,
            round(float(sc.x_semidim), 4),
            round(float(sc.y_semidim), 4),
            round(float(lane_spacing), 4),
            land_counts,
        )
        cur_step_max = int(sc.step_count.max().item())
        reset_detected = cur_step_max < self._prev_step_max
        self._prev_step_max = cur_step_max
        if reset_detected:
            for cache in self.ground_route_cache:
                cache.clear()
        if self.drone_plan_signature == signature and not reset_detected:
            return

        plans: list[list[list[tuple[float, float]]]] = []
        for b in range(B):
            lanes = self._search_lanes_for_env(b, lane_spacing, footprint)
            blocks = self._balanced_lane_blocks(lanes, sc.n_drones)
            env_plans = []
            for drone_idx, block in enumerate(blocks):
                start_pos = sc.world.agents[drone_idx].state.pos[b]
                env_plans.append(self._waypoints_for_lane_block(block, start_pos))
            plans.append(env_plans)

        self.drone_waypoints = plans
        self.drone_waypoint_index = torch.zeros(B, sc.n_drones, dtype=torch.long, device=sc.fire_grid.device)
        self.drone_plan_signature = signature

    def _search_footprint(self) -> float:
        sc = self.scenario
        min_altitude = float(sc.drone_flight_levels.min().item())
        return max(min_altitude * sc.drone_camera_half_angle_tan, 1e-6)

    def _waypoint_tolerance(self, env_index: int) -> float:
        sc = self.scenario
        if hasattr(sc, "terrain_sim_units_per_meter"):
            sim_units_per_meter = float(sc.terrain_sim_units_per_meter[env_index].item())
            if sim_units_per_meter > 0.0:
                return max(DRONE_WAYPOINT_TOLERANCE_M * sim_units_per_meter, 1e-4)
        return 0.01 * sc.world_scale

    def _search_lanes_for_env(self, env_index: int, lane_spacing: float, footprint: float) -> list[dict]:
        sc = self.scenario
        size = sc.fire_grid_size
        device = sc.fire_grid.device
        margin = max(float(sc.agent_radius), 0.02 * sc.world_scale)
        x_min_world = -float(sc.x_semidim) + margin
        x_max_world = float(sc.x_semidim) - margin
        y_min_world = -float(sc.y_semidim) + margin
        y_max_world = float(sc.y_semidim) - margin
        span_y = max(y_max_world - y_min_world, 1e-6)
        n_lanes = max(1, int(math.ceil(span_y / lane_spacing)) + 1)
        lane_ys = torch.linspace(y_min_world, y_max_world, n_lanes, device=device)

        cell_w = 2.0 * float(sc.x_semidim) / size
        cell_h = 2.0 * float(sc.y_semidim) / size
        cell_xs = torch.linspace(
            -float(sc.x_semidim) + 0.5 * cell_w,
            float(sc.x_semidim) - 0.5 * cell_w,
            size,
            device=device,
        )
        cell_ys = torch.linspace(
            -float(sc.y_semidim) + 0.5 * cell_h,
            float(sc.y_semidim) - 0.5 * cell_h,
            size,
            device=device,
        )
        cover = sc.land_cover_grid[env_index]
        searchable = (cover != LAND_WATER) & (cover != LAND_ROCK)
        lanes: list[dict] = []
        for lane_y in lane_ys:
            y_mask = (cell_ys - lane_y).abs() <= lane_spacing * 0.5
            lane_mask = searchable & y_mask.view(size, 1)
            weight = int(lane_mask.sum().item())
            if weight <= 0:
                continue
            x_used = lane_mask.any(dim=0)
            used_indices = x_used.nonzero(as_tuple=False).flatten()
            land_x_min = float(cell_xs[int(used_indices.min().item())].item())
            land_x_max = float(cell_xs[int(used_indices.max().item())].item())
            segment_margin = max(0.5 * footprint, 0.5 * cell_w)
            lanes.append({
                "y": float(lane_y.item()),
                "x_min": max(x_min_world, land_x_min - segment_margin),
                "x_max": min(x_max_world, land_x_max + segment_margin),
                "weight": float(weight),
            })
        if lanes:
            return lanes
        return [{
            "y": float(y.item()),
            "x_min": x_min_world,
            "x_max": x_max_world,
            "weight": 1.0,
        } for y in lane_ys]

    @staticmethod
    def _balanced_lane_blocks(lanes: list[dict], n_drones: int) -> list[list[dict]]:
        if n_drones <= 0:
            return []
        blocks: list[list[dict]] = []
        start = 0
        remaining_weight = sum(lane["weight"] for lane in lanes)
        for drone_idx in range(n_drones):
            remaining_drones = n_drones - drone_idx
            if start >= len(lanes):
                blocks.append([])
                continue
            if remaining_drones == 1:
                blocks.append(lanes[start:])
                break
            max_end = len(lanes) - (remaining_drones - 1)
            target = remaining_weight / remaining_drones if remaining_drones > 0 else remaining_weight
            acc = 0.0
            end = start
            while end < max_end:
                next_acc = acc + lanes[end]["weight"]
                if end > start and abs(next_acc - target) > abs(acc - target):
                    break
                acc = next_acc
                end += 1
            if end == start:
                acc = lanes[end]["weight"]
                end += 1
            blocks.append(lanes[start:end])
            remaining_weight -= acc
            start = end
        while len(blocks) < n_drones:
            blocks.append([])
        return blocks

    @staticmethod
    def _waypoints_for_lane_block(
        lanes: list[dict],
        start_pos: torch.Tensor,
    ) -> list[tuple[float, float]]:
        if not lanes:
            return []
        first_lane = lanes[0]
        left_start = torch.tensor([first_lane["x_min"], first_lane["y"]], device=start_pos.device)
        right_start = torch.tensor([first_lane["x_max"], first_lane["y"]], device=start_pos.device)
        start_left = bool((start_pos - left_start).norm().item() <= (start_pos - right_start).norm().item())
        waypoints: list[tuple[float, float]] = []
        for idx, lane in enumerate(lanes):
            left = (float(lane["x_min"]), float(lane["y"]))
            right = (float(lane["x_max"]), float(lane["y"]))
            first, second = (left, right) if (idx % 2 == 0) == start_left else (right, left)
            waypoints.extend((first, second))
        return waypoints


# ----------------------------------------------------------------------
# Nearest-candidate (drones random walk, ground -> nearest scouted)
# ----------------------------------------------------------------------
class NearestCandidatePolicy:
    """
    Drones use persistent random walks (no map coverage strategy).
    Ground robots head to the *nearest* survivor that has been scouted
    by any drone but not yet confirmed.

    This is the canonical "obvious heuristic" baseline that HAPPO should
    beat — it's reactive, doesn't reason about staleness or hazard.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        self._random_walk = RandomWalkPolicy(env)
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        out: List[torch.Tensor] = []

        # Drones: coherent but uninformed persistent random search.
        random_walk_actions = self._random_walk(env)
        rand_actions = env.get_random_actions()
        for i in range(sc.n_drones):
            out.append(random_walk_actions[i])

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
        self._prev_step_max = -1
        self._lawnmower = LawnmowerPolicy(env)
        self.ground_route_cache = [dict() for _ in range(self.scenario.n_ground)]

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        cur_step_max = int(sc.step_count.max().item())
        if cur_step_max < self._prev_step_max:
            self.scout_step.fill_(-1)
            for cache in self.ground_route_cache:
                cache.clear()
        self._prev_step_max = cur_step_max

        newly_scouted = sc.scouted_survivors & (self.scout_step < 0)
        self.scout_step = torch.where(
            newly_scouted,
            torch.full_like(self.scout_step, float(self.t)),
            self.scout_step,
        )

        actions = self._lawnmower(env)

        # Replace ground actions with freshest unassigned scouted targets.
        # Unassigned ground robots fall back to random (not lawnmower) so they
        # explore rather than follow an aerial coverage path.
        found = sc.found_survivors
        targetable = (self.scout_step >= 0) & ~found
        random_actions = env.get_random_actions()
        fallback = [
            random_actions[sc.n_drones + gi]
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
# Ant-colony-inspired distributed recency maps
# ----------------------------------------------------------------------
class AntColonyPolicy:
    """Distributed stigmergic coverage with dropout-sensitive local memory.

    Each drone stores the last observation step for every map cell and
    deposits timestamps over its physical camera footprint. A receiver with
    live communications merges the newest timestamps and survivor events from
    the team; during dropout it retains that stale map and adds only its own
    observations. Drones move toward the nearest least-recently-seen searchable
    cell. UGVs route to survivors known in their own local event memory.
    """

    def __init__(self, env):
        self.scenario: WildfireSearchScenario = env.scenario
        sc = self.scenario
        B = sc.world.batch_dim
        A = len(sc.world.agents)
        G = sc.fire_grid_size
        device = sc.fire_grid.device

        self.last_seen = torch.full(
            (B, sc.n_drones, G, G),
            PHEROMONE_UNSEEN_STEP,
            dtype=torch.long,
            device=device,
        )
        self.known_survivors = torch.zeros(
            B, A, sc.n_survivors, dtype=torch.bool, device=device,
        )
        self.known_confirmed = torch.zeros_like(self.known_survivors)
        self.drone_targets = torch.full(
            (B, sc.n_drones, 2), -1, dtype=torch.long, device=device,
        )
        self.target_assignment_step = torch.full(
            (B, sc.n_drones), -1, dtype=torch.long, device=device,
        )
        self.previous_step_count = sc.step_count.clone()
        self.ground_route_cache = [dict() for _ in range(sc.n_ground)]

        cell_w = 2.0 * float(sc.x_semidim) / G
        cell_h = 2.0 * float(sc.y_semidim) / G
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.grid_xs = torch.linspace(
            -float(sc.x_semidim) + cell_w / 2.0,
            float(sc.x_semidim) - cell_w / 2.0,
            G,
            device=device,
        )
        self.grid_ys = torch.linspace(
            -float(sc.y_semidim) + cell_h / 2.0,
            float(sc.y_semidim) - cell_h / 2.0,
            G,
            device=device,
        )

    def __call__(self, env) -> List[torch.Tensor]:
        del env
        self._reset_finished_environments()
        self._deposit_local_observations()
        self._merge_connected_memories()

        actions = self._drone_actions()
        actions.extend(self._ground_actions())
        self.previous_step_count = self.scenario.step_count.clone()
        return actions

    def _reset_finished_environments(self) -> None:
        sc = self.scenario
        reset = sc.step_count < self.previous_step_count
        for b in reset.nonzero(as_tuple=False).flatten().tolist():
            self.last_seen[b].fill_(PHEROMONE_UNSEEN_STEP)
            self.known_survivors[b].zero_()
            self.known_confirmed[b].zero_()
            self.drone_targets[b].fill_(-1)
            self.target_assignment_step[b].fill_(-1)
            for cache in self.ground_route_cache:
                cache.pop(int(b), None)

    def _deposit_local_observations(self) -> None:
        sc = self.scenario
        if sc.n_drones > 0:
            drone_pos = torch.stack(
                [agent.state.pos for agent in sc.world.agents[:sc.n_drones]],
                dim=1,
            )
            dx = (
                (self.grid_xs.view(1, 1, 1, -1) - drone_pos[..., X].view(
                    sc.world.batch_dim, sc.n_drones, 1, 1,
                )).abs()
                - self.cell_w / 2.0
            ).clamp_min(0.0)
            dy = (
                (self.grid_ys.view(1, 1, -1, 1) - drone_pos[..., Y].view(
                    sc.world.batch_dim, sc.n_drones, 1, 1,
                )).abs()
                - self.cell_h / 2.0
            ).clamp_min(0.0)
            footprint = sc._drone_camera_ranges().view(
                sc.world.batch_dim, sc.n_drones, 1, 1,
            )
            observed = dx.square() + dy.square() <= footprint.square()
            step = sc.step_count.view(-1, 1, 1, 1).expand_as(self.last_seen)
            self.last_seen = torch.where(observed, step, self.last_seen)

            detections = getattr(sc, "step_drone_detections", None)
            if detections is not None:
                self.known_survivors[:, :sc.n_drones] |= detections

        confirmations = getattr(sc, "step_ground_confirmations", None)
        if confirmations is not None and sc.n_ground > 0:
            self.known_confirmed[:, sc.n_drones:] |= confirmations
            self.known_survivors[:, sc.n_drones:] |= confirmations

    def _communication_mask(self) -> torch.Tensor:
        sc = self.scenario
        flags = []
        for agent in sc.world.agents:
            comms_up = getattr(agent, "comms_up", None)
            if comms_up is None:
                comms_up = torch.ones(
                    sc.world.batch_dim, dtype=torch.bool, device=sc.fire_grid.device,
                )
            flags.append(comms_up)
        if not flags:
            return torch.zeros(
                sc.world.batch_dim, 0, dtype=torch.bool, device=sc.fire_grid.device,
            )
        return torch.stack(flags, dim=1)

    def _merge_connected_memories(self) -> None:
        sc = self.scenario
        comms_up = self._communication_mask()
        if sc.n_drones > 0:
            self.last_seen = _merge_local_timestamp_maps(
                self.last_seen,
                comms_up[:, :sc.n_drones],
            )
        self.known_survivors = _merge_local_bool_knowledge(
            self.known_survivors,
            comms_up,
        )
        self.known_confirmed = _merge_local_bool_knowledge(
            self.known_confirmed,
            comms_up,
        )

    def _drone_actions(self) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        device = sc.fire_grid.device
        searchable = (sc.land_cover_grid != LAND_WATER) & (sc.land_cover_grid != LAND_ROCK)
        actions: List[torch.Tensor] = []

        for drone_idx in range(sc.n_drones):
            pos = sc.world.agents[drone_idx].state.pos
            action = torch.zeros(B, 2, device=device)
            for b in range(B):
                target = self.drone_targets[b, drone_idx]
                target_valid = bool((target >= 0).all().item())
                if target_valid:
                    tx, ty = int(target[X].item()), int(target[Y].item())
                    target_world = _grid_to_world(sc, tx, ty, device)
                    reached = float((target_world - pos[b]).norm().item()) <= max(
                        0.5 * sc._drone_camera_ranges()[b, drone_idx].item(),
                        max(self.cell_w, self.cell_h),
                    )
                    remotely_searched = (
                        int(self.last_seen[b, drone_idx, ty, tx].item())
                        > int(self.target_assignment_step[b, drone_idx].item())
                    )
                    target_valid = not reached and not remotely_searched

                if not target_valid:
                    tx, ty = self._select_drone_target(b, drone_idx, searchable[b], pos[b])
                    self.drone_targets[b, drone_idx] = torch.tensor(
                        [tx, ty], dtype=torch.long, device=device,
                    )
                    self.target_assignment_step[b, drone_idx] = sc.step_count[b]

                tx = int(self.drone_targets[b, drone_idx, X].item())
                ty = int(self.drone_targets[b, drone_idx, Y].item())
                target_world = _grid_to_world(sc, tx, ty, device)
                delta = target_world - pos[b]
                distance = float(delta.norm().item())
                direction = delta / delta.norm().clamp_min(1e-6)
                slowdown = max(
                    float(sc._drone_camera_ranges()[b, drone_idx].item()),
                    max(self.cell_w, self.cell_h),
                )
                action[b] = _drone_arrival_action(
                    direction=direction,
                    distance=distance,
                    velocity=sc.world.agents[drone_idx].state.vel[b],
                    max_speed=float(sc.drone_max_speed_sim),
                    slowdown_distance=slowdown,
                )
            actions.append(action)
        return actions

    def _select_drone_target(
        self,
        env_index: int,
        drone_index: int,
        searchable: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[int, int]:
        timestamps = self.last_seen[env_index, drone_index]
        available = searchable
        if not bool(available.any().item()):
            gx = int(
                ((position[X] + self.scenario.x_semidim)
                 / (2 * self.scenario.x_semidim)
                 * self.scenario.fire_grid_size)
                .clamp(0, self.scenario.fire_grid_size - 1)
                .item()
            )
            gy = int(
                ((position[Y] + self.scenario.y_semidim)
                 / (2 * self.scenario.y_semidim)
                 * self.scenario.fire_grid_size)
                .clamp(0, self.scenario.fire_grid_size - 1)
                .item()
            )
            return gx, gy

        oldest = timestamps[available].min()
        candidates = available & (timestamps == oldest)
        distance_sq = (
            (self.grid_xs.view(1, -1) - position[X]).square()
            + (self.grid_ys.view(-1, 1) - position[Y]).square()
        )
        distance_sq = torch.where(
            candidates,
            distance_sq,
            torch.full_like(distance_sq, float("inf")),
        )
        flat = int(distance_sq.argmin().item())
        return flat % self.scenario.fire_grid_size, flat // self.scenario.fire_grid_size

    def _ground_actions(self) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        device = sc.fire_grid.device
        if sc.n_ground == 0:
            return []

        survivor_pos = torch.stack([s.state.pos for s in sc._survivors], dim=1)
        comms_up = self._communication_mask()
        selected = torch.full(
            (B, sc.n_ground), -1, dtype=torch.long, device=device,
        )
        actions: List[torch.Tensor] = []
        batch_idx = torch.arange(B, device=device)

        for gi in range(sc.n_ground):
            agent_idx = sc.n_drones + gi
            position = sc.world.agents[agent_idx].state.pos
            available = (
                self.known_survivors[:, agent_idx]
                & ~self.known_confirmed[:, agent_idx]
            )
            for previous in range(gi):
                mutually_connected = comms_up[:, agent_idx] & comms_up[:, sc.n_drones + previous]
                previous_target = selected[:, previous]
                valid_previous = mutually_connected & (previous_target >= 0)
                if valid_previous.any():
                    available[
                        batch_idx[valid_previous],
                        previous_target[valid_previous],
                    ] = False

            any_target = available.any(dim=-1)
            distances = (survivor_pos - position.unsqueeze(1)).norm(dim=-1)
            scores = torch.where(
                available,
                distances,
                torch.full_like(distances, float("inf")),
            )
            target_idx = scores.argmin(dim=-1)
            selected[:, gi] = torch.where(
                any_target,
                target_idx,
                torch.full_like(target_idx, -1),
            )
            target_pos = survivor_pos.gather(
                1, target_idx.view(B, 1, 1).expand(B, 1, 2),
            ).squeeze(1)
            waypoint = _route_ground_waypoints(
                sc,
                gi,
                target_pos,
                target_idx,
                self.ground_route_cache,
            )
            hold = torch.zeros(B, 2, device=device)
            move = _terrain_safe_ground_action(sc, gi, waypoint, hold)
            actions.append(torch.where(any_target.unsqueeze(-1), move, hold))
        return actions


# ----------------------------------------------------------------------
# Registry — for use from scripts/notebooks
# ----------------------------------------------------------------------
BASELINES: dict[str, Callable] = {
    "random_action":       RandomActionPolicy,
    "random_walk":         RandomWalkPolicy,
    "lawnmower":           LawnmowerPolicy,
    "nearest_candidate":   NearestCandidatePolicy,
    "highest_confidence":  HighestConfidencePolicy,
    "ant_colony":          AntColonyPolicy,
}


def get_baseline(name: str, env) -> Callable:
    """Resolve a baseline by name, constructing it on the given env."""
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline {name!r}. Available: {list(BASELINES)}")
    cls = BASELINES[name]
    # RandomActionPolicy doesn't need env, others do
    return cls(env) if cls is not RandomActionPolicy else cls()

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

from typing import Callable, List

import torch

from envs.wildfire_search import WildfireSearchScenario, X, Y


# Each policy returns a list of (B, action_dim) action tensors, one per
# agent, in the same order as env.agents. WildfireSearchScenario's
# actions are 2D continuous in [-1, 1] (a force vector).


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

        # ---- Ground robots: go to nearest scouted survivor ----
        scouted = sc.scouted_survivors        # (B, S) bool
        found   = sc.found_survivors          # (B, S) bool
        targetable = scouted & ~found         # not yet confirmed

        # Survivor positions (B, S, 2)
        surv_pos = torch.stack([s.state.pos for s in sc._survivors], dim=1)

        for gi in range(sc.n_ground):
            ag_idx = sc.n_drones + gi
            pos = sc.world.agents[ag_idx].state.pos  # (B, 2)
            # Distance to each survivor (B, S)
            d = (surv_pos - pos.unsqueeze(1)).norm(dim=-1)
            d_masked = torch.where(targetable, d, torch.full_like(d, float("inf")))
            best = d_masked.argmin(dim=-1)               # (B,)
            target_pos = surv_pos.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            # If no targetable survivor, hold still
            any_targetable = targetable.any(dim=-1)
            delta = (target_pos - pos).clamp(-1.0, 1.0)
            delta = torch.where(any_targetable.unsqueeze(-1), delta, torch.zeros_like(delta))
            out.append(delta)

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

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim
        out: List[torch.Tensor] = []

        # Drones: random
        rand_actions = env.get_random_actions()
        for i in range(sc.n_drones):
            out.append(rand_actions[i])

        # Ground: nearest scouted survivor
        scouted    = sc.scouted_survivors
        found      = sc.found_survivors
        targetable = scouted & ~found
        surv_pos   = torch.stack([s.state.pos for s in sc._survivors], dim=1)

        for gi in range(sc.n_ground):
            ag_idx = sc.n_drones + gi
            pos = sc.world.agents[ag_idx].state.pos
            d = (surv_pos - pos.unsqueeze(1)).norm(dim=-1)
            d_masked = torch.where(targetable, d, torch.full_like(d, float("inf")))
            best = d_masked.argmin(dim=-1)
            target_pos = surv_pos.gather(1, best.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            any_t = targetable.any(dim=-1)
            delta = (target_pos - pos).clamp(-1.0, 1.0)
            delta = torch.where(any_t.unsqueeze(-1), delta, rand_actions[ag_idx])
            out.append(delta)

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
        self.scout_step: List[int] = [-1] * self.scenario.n_survivors
        self.t = 0
        self._lawnmower = LawnmowerPolicy(env)

    def __call__(self, env) -> List[torch.Tensor]:
        sc = self.scenario
        B = sc.world.batch_dim

        # Update scout step tracking (env_index=0 for control logic)
        sc_mask = sc.scouted_survivors[0].cpu().tolist()
        for i, s in enumerate(sc_mask):
            if s and self.scout_step[i] < 0:
                self.scout_step[i] = self.t

        actions = self._lawnmower(env)

        # Replace ground actions to use freshest scouted as target
        found = sc.found_survivors
        surv_pos = torch.stack([s.state.pos for s in sc._survivors], dim=1)

        for gi in range(sc.n_ground):
            ag_idx = sc.n_drones + gi
            pos = sc.world.agents[ag_idx].state.pos

            # Build per-survivor freshness score; not-yet-scouted → -inf
            freshness = torch.tensor(
                [self.scout_step[i] if self.scout_step[i] >= 0 else -1
                 for i in range(sc.n_survivors)],
                device=pos.device, dtype=torch.float,
            )
            # Mask out already-found ones
            valid = (freshness >= 0) & ~found[0]
            if valid.any():
                # Pick the freshest one (highest scout_step)
                freshness_masked = torch.where(
                    valid, freshness, torch.full_like(freshness, float("-inf")),
                )
                target_idx = freshness_masked.argmax().item()
                tgt_pos = surv_pos[:, target_idx, :]
                delta = (tgt_pos - pos).clamp(-1.0, 1.0)
                actions[ag_idx] = delta

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

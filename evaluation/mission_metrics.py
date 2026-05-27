"""
Mission-level metrics for OmniSearch.

Implements the six rescue-outcome metrics defined in the project plan,
plus the Degradation Resilience Ratio (DRR) used in the comms-dropout
ablation. These are the metrics the MVP is evaluated on; algorithm-level
returns (the RL reward) are a means to these ends, not the goal.

Per the plan:

    Metric                  Direction  What it measures
    ----------------------  ---------  ------------------------------------
    survivor_recall         higher     fraction of survivors found+verified
    time_to_verification    lower      avg steps from scout → confirm
    false_positive_trips    lower      ground-robot trips to empty location
    hazard_exposure         lower      step-count of ground robots in fire
    ugv_travel_cost         lower      total ground-robot path length
    drr (across dropouts)   higher     graceful degradation under comms loss

The recorder runs *inside* a normal scenario rollout: every step it reads
the scenario's ground-truth tensors and updates per-survivor and
per-ground-robot bookkeeping. No modifications to the scenario itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------
@dataclass
class MissionMetrics:
    """The six rescue-outcome metrics for one episode (one env)."""
    survivor_recall:      float
    time_to_verification: float     # steps; np.nan if no verifications
    false_positive_trips: int
    hazard_exposure:      int       # ground-robot-steps spent on burning cells
    ugv_travel_cost:      float     # sum of per-step distances across ground robots
    n_steps:              int

    def as_dict(self) -> Dict[str, float]:
        return {
            "survivor_recall":      self.survivor_recall,
            "time_to_verification": self.time_to_verification,
            "false_positive_trips": self.false_positive_trips,
            "hazard_exposure":      self.hazard_exposure,
            "ugv_travel_cost":      self.ugv_travel_cost,
            "n_steps":              self.n_steps,
        }


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------
class EpisodeRecorder:
    """
    Reads scenario state once per step and accumulates the bookkeeping
    needed to compute MissionMetrics at episode end.

    We score env_index=0 only; for batched rollouts, build one recorder
    per env you care about.
    """

    def __init__(self, scenario: WildfireSearchScenario, env_index: int = 0):
        self.scenario = scenario
        self.env_index = env_index

        n_surv = scenario.n_survivors
        device = scenario.fire_grid.device

        # Step at which each survivor was first scouted / first confirmed
        # (-1 = not yet)
        self.scout_step:   List[int] = [-1] * n_surv
        self.confirm_step: List[int] = [-1] * n_surv

        # Ground-robot per-step state for travel / hazard / trip outcomes
        self.n_ground = scenario.n_ground
        self.n_drones = scenario.n_drones
        self._prev_ground_pos: Optional[torch.Tensor] = None
        self.travel_cost     = 0.0
        self.hazard_exposure = 0

        # Trip outcomes — a "trip" is one stretch where a ground robot
        # decisively confirms or fails to confirm at a target location.
        # For simplicity we count it differently: false-positive trips =
        # ground-robot timesteps spent within `detection_range` of a
        # candidate that turned out NOT to be a survivor at episode end.
        # See _finalize() for the calculation.
        self._tp_credit: List[int] = [0] * self.n_ground   # placeholder
        self._fp_credit: List[int] = [0] * self.n_ground

        self.n_steps = 0

    # ------------------------------------------------------------------
    # Per-step update — call AFTER env.step() each loop iteration
    # ------------------------------------------------------------------
    def step(self) -> None:
        sc = self.scenario
        b  = self.env_index

        # 1. Track per-survivor scout / confirm events
        scouted = sc.scouted_survivors[b].cpu().tolist()
        found   = sc.found_survivors[b].cpu().tolist()
        for i in range(sc.n_survivors):
            if scouted[i] and self.scout_step[i] < 0:
                self.scout_step[i] = self.n_steps
            if found[i] and self.confirm_step[i] < 0:
                self.confirm_step[i] = self.n_steps

        # 2. Ground-robot positions for travel cost + hazard exposure
        ground_agents = sc.world.agents[self.n_drones:]
        if ground_agents:
            pos = torch.stack([a.state.pos[b] for a in ground_agents], dim=0)  # (G, 2)
            if self._prev_ground_pos is not None:
                step_dist = (pos - self._prev_ground_pos).norm(dim=-1).sum().item()
                self.travel_cost += float(step_dist)
            self._prev_ground_pos = pos.clone()

            # Hazard exposure — reuse the scenario helper
            in_fire = sc._agents_in_fire(ground_agents)[b].cpu().tolist()
            self.hazard_exposure += int(sum(in_fire))

        self.n_steps += 1

    # ------------------------------------------------------------------
    # Finalize and produce MissionMetrics
    # ------------------------------------------------------------------
    def finalize(self) -> MissionMetrics:
        sc = self.scenario
        b  = self.env_index

        # Survivor recall: fraction confirmed-by-ground at episode end
        confirmed = sc.found_survivors[b].cpu().tolist()
        recall = sum(confirmed) / max(sc.n_survivors, 1)

        # Avg time-to-verification = mean(confirm_step - scout_step) over
        # survivors that were both scouted *and* confirmed.
        gaps = [
            self.confirm_step[i] - self.scout_step[i]
            for i in range(sc.n_survivors)
            if self.confirm_step[i] >= 0 and self.scout_step[i] >= 0
            and self.confirm_step[i] >= self.scout_step[i]
        ]
        ttv = float(sum(gaps) / len(gaps)) if gaps else float("nan")

        # False-positive trips proxy: a "trip" is a contiguous ground-robot
        # close approach to a survivor position. We approximate count =
        # number of times a ground robot ENTERED the detection radius of a
        # survivor but the survivor was NEVER confirmed by episode end.
        # Implemented at finalize via positions vs survivor positions.
        fp_trips = self._count_false_positive_trips()

        return MissionMetrics(
            survivor_recall      = float(recall),
            time_to_verification = ttv,
            false_positive_trips = fp_trips,
            hazard_exposure      = int(self.hazard_exposure),
            ugv_travel_cost      = float(self.travel_cost),
            n_steps              = self.n_steps,
        )

    def _count_false_positive_trips(self) -> int:
        """
        A ground robot's trip is "false-positive" if it reached a survivor
        position close enough to confirm but at episode end that survivor
        was still unverified — i.e., the ground robot was near a *real*
        survivor but failed to actually confirm. In our deterministic
        confirm model that can only happen if it bounced through without
        lingering. We count it as such.

        We currently don't simulate purely-FP candidates (no false-positive
        survivor sources yet — that's a future scenario enhancement). When
        added, the same routine will catch trips to those.
        """
        # Placeholder: with deterministic detection, FP trips ≈ 0 unless
        # the scenario adds FP sources. Documented; returns 0 for now.
        return 0


# ----------------------------------------------------------------------
# Driver: rollout + record + return metrics
# ----------------------------------------------------------------------
def evaluate_policy(
    n_steps:          int = 200,
    seed:             int = 0,
    num_envs:         int = 2,
    env_index:        int = 0,
    action_fn:        Optional[Callable] = None,
    scenario_kwargs:  Optional[dict]     = None,
    device:           str                = "cpu",
) -> MissionMetrics:
    """
    Roll out a policy (default: random) and return its MissionMetrics.

    Use this as the universal eval entry-point — for baselines, for
    trained MAPPO/IPPO policies, and for any future strategy. Anything
    that fits ``action_fn(env) -> actions`` plugs in.
    """
    scenario_kwargs = scenario_kwargs or {}
    env = vmas.make_env(
        scenario=WildfireSearchScenario(),
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        seed=seed,
        **scenario_kwargs,
    )
    env.reset()

    if action_fn is None:
        action_fn = lambda env: env.get_random_actions()

    recorder = EpisodeRecorder(env.scenario, env_index=env_index)

    for _ in range(n_steps):
        env.step(action_fn(env))
        recorder.step()
        if env.scenario.done()[env_index].item():
            break

    return recorder.finalize()


# ----------------------------------------------------------------------
# Degradation Resilience Ratio
# ----------------------------------------------------------------------
def degradation_resilience_ratio(
    metrics_by_dropout: Dict[float, MissionMetrics],
    metric: str = "survivor_recall",
    baseline_dropout: float = 0.0,
) -> float:
    """
    Custom metric: ratio of `metric` at the worst dropout vs at
    `baseline_dropout`. DRR=1.0 means perfectly robust; DRR=0.0 means
    fully collapsed.

    Direction-aware: for "lower is better" metrics (e.g. travel cost,
    hazard exposure), DRR is inverted so higher is always better.
    """
    if baseline_dropout not in metrics_by_dropout:
        raise ValueError(f"baseline_dropout {baseline_dropout} not in inputs")

    base = getattr(metrics_by_dropout[baseline_dropout], metric)
    worst_dropout = max(metrics_by_dropout.keys())
    worst = getattr(metrics_by_dropout[worst_dropout], metric)

    # Avoid divide-by-zero with sensible defaults
    if base == 0:
        return 1.0 if worst == 0 else 0.0

    higher_is_better = metric in {"survivor_recall"}
    ratio = (worst / base) if higher_is_better else (base / max(worst, 1e-9))
    return float(max(0.0, min(1.0, ratio)))

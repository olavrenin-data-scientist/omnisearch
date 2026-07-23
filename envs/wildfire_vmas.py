"""VMAS construction helpers for the wildfire scenario.

VMAS does not pass ``Environment.reset(seed=...)``'s seed argument to a
scenario.  The reset-aware environment below supplies that missing lifecycle
signal without changing VMAS itself or consuming global random numbers.
"""

from __future__ import annotations

from typing import Optional, Union

from vmas import scenarios
from vmas.simulator.environment import Environment, Wrapper
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import DEVICE_TYPING


class WildfireEnvironment(Environment):
    """VMAS environment that reports explicit reseeds to its scenario."""

    def _seed(self, seed=None):
        result = super()._seed(seed=seed)
        notify = getattr(self.scenario, "_notify_explicit_reset_seed", None)
        if notify is not None:
            notify(0 if seed is None else int(seed))
        return result


def make_wildfire_env(
    scenario: Union[str, BaseScenario],
    num_envs: int,
    device: DEVICE_TYPING = "cpu",
    continuous_actions: bool = True,
    wrapper: Optional[Union[Wrapper, str]] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    dict_spaces: bool = False,
    multidiscrete_actions: bool = False,
    clamp_actions: bool = False,
    grad_enabled: bool = False,
    terminated_truncated: bool = False,
    wrapper_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Construct a wildfire VMAS environment with reset-seed handoff."""
    if isinstance(scenario, str):
        if not scenario.endswith(".py"):
            scenario += ".py"
        scenario = scenarios.load(scenario).Scenario()

    env = WildfireEnvironment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        seed=seed,
        dict_spaces=dict_spaces,
        multidiscrete_actions=multidiscrete_actions,
        clamp_actions=clamp_actions,
        grad_enabled=grad_enabled,
        terminated_truncated=terminated_truncated,
        **kwargs,
    )

    if wrapper is not None and isinstance(wrapper, str):
        wrapper = Wrapper[wrapper.upper()]
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    return wrapper.get_env(env, **wrapper_kwargs) if wrapper is not None else env

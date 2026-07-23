import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
import vmas

from envs.wildfire_search import WildfireSearchScenario, _ResetRngContext


ROOT = Path(__file__).resolve().parent.parent
TERRAIN_CACHE = ROOT / "data" / "terrain_cache" / "malibu_creek_500m_128.npz"


@dataclass(frozen=True)
class _ResetSnapshot:
    active_survivors: torch.Tensor
    active_decoys: torch.Tensor
    survivor_positions: torch.Tensor
    decoy_positions: torch.Tensor
    survivor_reveal_steps: torch.Tensor
    decoy_reveal_steps: torch.Tensor
    uav_positions: torch.Tensor
    ugv_positions: torch.Tensor


class ScenarioRngInvarianceTests(unittest.TestCase):
    schema_n_drones = 3
    schema_n_ground = 2

    def _env(
        self,
        *,
        seed: int,
        n_drones: int,
        n_ground: int,
        num_envs: int = 1,
        disable_fire: bool = True,
    ):
        return WildfireSearchScenario.make_env(
            scenario=WildfireSearchScenario(),
            num_envs=num_envs,
            device="cpu",
            continuous_actions=True,
            seed=seed,
            n_drones=n_drones,
            n_ground=n_ground,
            n_survivors=8,
            n_decoys=3,
            obs_schema_n_drones=self.schema_n_drones,
            obs_schema_n_ground=self.schema_n_ground,
            obs_schema_n_survivors=8,
            active_survivors_min=3,
            active_survivors_max=8,
            active_decoys_min=0,
            active_decoys_max=3,
            known_survivors_at_reset=False,
            delayed_survivor_knowledge=True,
            survivor_reveal_initial_count=1,
            survivor_reveal_start_step=10,
            survivor_reveal_end_step=180,
            delayed_decoy_knowledge=True,
            decoy_reveal_initial_count=0,
            decoy_reveal_start_step=10,
            decoy_reveal_end_step=180,
            disable_fire=disable_fire,
            fire_grid_size=128,
            max_steps=300,
            terrain_source="real",
            terrain_cache_path=str(TERRAIN_CACHE),
            uav_start_min_separation_m=150.0,
            uav_start_edge_margin_m=50.0,
        )

    @staticmethod
    def _entity_positions(entities, env_index: int) -> torch.Tensor:
        if not entities:
            return torch.empty(0, 2, dtype=torch.float32)
        return torch.stack(
            [entity.state.pos[env_index].detach().cpu().clone() for entity in entities],
            dim=0,
        )

    def _snapshot(self, scenario, env_index: int) -> _ResetSnapshot:
        drone_agents = scenario.world.agents[: scenario.n_drones]
        ground_agents = scenario.world.agents[
            scenario.n_drones : scenario.n_drones + scenario.n_ground
        ]
        return _ResetSnapshot(
            active_survivors=scenario.active_survivors[env_index].detach().cpu().clone(),
            active_decoys=scenario.active_decoys[env_index].detach().cpu().clone(),
            survivor_positions=self._entity_positions(scenario._survivors, env_index),
            decoy_positions=self._entity_positions(scenario._decoys, env_index),
            survivor_reveal_steps=scenario.survivor_reveal_steps[env_index]
            .detach()
            .cpu()
            .clone(),
            decoy_reveal_steps=scenario.decoy_reveal_steps[env_index]
            .detach()
            .cpu()
            .clone(),
            uav_positions=self._entity_positions(drone_agents, env_index),
            ugv_positions=self._entity_positions(ground_agents, env_index),
        )

    def _reset_snapshot(
        self,
        *,
        seed: int,
        n_drones: int,
        n_ground: int,
    ) -> _ResetSnapshot:
        env = self._env(seed=seed, n_drones=n_drones, n_ground=n_ground)
        env.reset(seed=seed)
        return self._snapshot(env.scenario, 0)

    def _assert_mission_snapshot_equal(
        self,
        actual: _ResetSnapshot,
        expected: _ResetSnapshot,
    ) -> None:
        self.assertTrue(torch.equal(actual.active_survivors, expected.active_survivors))
        self.assertTrue(torch.equal(actual.active_decoys, expected.active_decoys))
        self.assertTrue(torch.equal(actual.survivor_positions, expected.survivor_positions))
        self.assertTrue(torch.equal(actual.decoy_positions, expected.decoy_positions))
        self.assertTrue(
            torch.equal(actual.survivor_reveal_steps, expected.survivor_reveal_steps)
        )
        self.assertTrue(torch.equal(actual.decoy_reveal_steps, expected.decoy_reveal_steps))

    def _assert_snapshot_equal(
        self,
        actual: _ResetSnapshot,
        expected: _ResetSnapshot,
    ) -> None:
        self._assert_mission_snapshot_equal(actual, expected)
        self.assertTrue(torch.equal(actual.uav_positions, expected.uav_positions))
        self.assertTrue(torch.equal(actual.ugv_positions, expected.ugv_positions))

    def test_physical_agent_count_does_not_change_mission_snapshot(self):
        configurations = ((2, 2), (3, 1), (1, 1))
        for seed in (1000, 1001, 1013):
            expected = self._reset_snapshot(seed=seed, n_drones=3, n_ground=2)
            for n_drones, n_ground in configurations:
                with self.subTest(seed=seed, n_drones=n_drones, n_ground=n_ground):
                    actual = self._reset_snapshot(
                        seed=seed,
                        n_drones=n_drones,
                        n_ground=n_ground,
                    )
                    self._assert_mission_snapshot_equal(actual, expected)

    def test_retained_agent_starts_are_prefix_stable(self):
        configurations = ((2, 2), (3, 1), (1, 1))
        for seed in (1000, 1001, 1013):
            expected = self._reset_snapshot(seed=seed, n_drones=3, n_ground=2)
            for n_drones, n_ground in configurations:
                with self.subTest(seed=seed, n_drones=n_drones, n_ground=n_ground):
                    actual = self._reset_snapshot(
                        seed=seed,
                        n_drones=n_drones,
                        n_ground=n_ground,
                    )
                    self.assertTrue(
                        torch.equal(actual.uav_positions, expected.uav_positions[:n_drones])
                    )
                    self.assertTrue(
                        torch.equal(actual.ugv_positions, expected.ugv_positions[:n_ground])
                    )

    def test_same_seed_repeats_exact_snapshot(self):
        first = self._reset_snapshot(seed=1000, n_drones=3, n_ground=2)
        second = self._reset_snapshot(seed=1000, n_drones=3, n_ground=2)
        self._assert_snapshot_equal(first, second)

    def test_different_seed_changes_snapshot(self):
        first = self._reset_snapshot(seed=1000, n_drones=3, n_ground=2)
        second = self._reset_snapshot(seed=1001, n_drones=3, n_ground=2)
        fields = (
            "active_survivors",
            "active_decoys",
            "survivor_positions",
            "decoy_positions",
            "survivor_reveal_steps",
            "decoy_reveal_steps",
            "uav_positions",
            "ugv_positions",
        )
        self.assertTrue(
            any(not torch.equal(getattr(first, field), getattr(second, field)) for field in fields)
        )

    def test_asynchronous_reset_order_does_not_change_next_episode(self):
        first = self._env(seed=1000, n_drones=3, n_ground=2, num_envs=2)
        second = self._env(seed=1000, n_drones=3, n_ground=2, num_envs=2)
        first.reset(seed=1000)
        second.reset(seed=1000)

        first.reset_at(index=0, return_observations=False)
        first.reset_at(index=1, return_observations=False)
        second.reset_at(index=1, return_observations=False)
        second.reset_at(index=0, return_observations=False)

        for env_index in (0, 1):
            with self.subTest(env_index=env_index):
                self._assert_snapshot_equal(
                    self._snapshot(first.scenario, env_index),
                    self._snapshot(second.scenario, env_index),
                )

    @staticmethod
    def _initial_fire_snapshot(scenario, env_index: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor[env_index].detach().cpu().clone()
            for tensor in (
                scenario.fire_grid,
                scenario.burned_grid,
                scenario.fire_age_grid,
                scenario.fire_lifetime_grid,
                scenario.fire_intensity_grid,
            )
        )

    def test_initial_fire_does_not_depend_on_physical_agent_count(self):
        configurations = ((3, 2), (2, 2), (3, 1), (1, 1))
        for seed in (1000, 1001, 1013):
            expected = None
            for n_drones, n_ground in configurations:
                env = self._env(
                    seed=seed,
                    n_drones=n_drones,
                    n_ground=n_ground,
                    disable_fire=False,
                )
                env.reset(seed=seed)
                actual = self._initial_fire_snapshot(env.scenario, 0)
                if expected is None:
                    expected = actual
                else:
                    for actual_tensor, expected_tensor in zip(actual, expected):
                        self.assertTrue(torch.equal(actual_tensor, expected_tensor))

    def test_initial_fire_is_stable_across_asynchronous_reset_order(self):
        first = self._env(
            seed=1000,
            n_drones=3,
            n_ground=2,
            num_envs=2,
            disable_fire=False,
        )
        second = self._env(
            seed=1000,
            n_drones=3,
            n_ground=2,
            num_envs=2,
            disable_fire=False,
        )
        first.reset(seed=1000)
        second.reset(seed=1000)

        first.reset_at(index=0, return_observations=False)
        first.reset_at(index=1, return_observations=False)
        second.reset_at(index=1, return_observations=False)
        second.reset_at(index=0, return_observations=False)

        for env_index in (0, 1):
            first_fire = self._initial_fire_snapshot(first.scenario, env_index)
            second_fire = self._initial_fire_snapshot(second.scenario, env_index)
            for first_tensor, second_tensor in zip(first_fire, second_fire):
                self.assertTrue(torch.equal(first_tensor, second_tensor))

    def test_initial_fire_seed_changes_with_scenario_seed(self):
        snapshots = []
        for seed in (1000, 1001):
            env = self._env(
                seed=seed,
                n_drones=3,
                n_ground=2,
                disable_fire=False,
            )
            env.reset(seed=seed)
            snapshots.append(self._initial_fire_snapshot(env.scenario, 0))

        self.assertTrue(
            any(
                not torch.equal(first, second)
                for first, second in zip(snapshots[0], snapshots[1])
            )
        )

    def test_initial_fire_sampling_does_not_advance_global_rng(self):
        env = self._env(
            seed=1000,
            n_drones=3,
            n_ground=2,
            disable_fire=True,
        )
        env.reset(seed=1000)
        scenario = env.scenario
        scenario.fire_grid.zero_()
        scenario.burned_grid.zero_()
        scenario.fire_age_grid.zero_()
        scenario.fire_lifetime_grid.zero_()
        scenario.fire_intensity_grid.zero_()
        scenario._reset_rng_contexts[0] = _ResetRngContext(
            base_seed=1000,
            env_index=0,
            episode_index=0,
        )
        torch.manual_seed(2468)
        state_before = torch.random.get_rng_state().clone()

        scenario._seed_initial_fire(0, scenario.fire_grid_size, scenario.fire_grid_size)

        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))

    def test_cell_sampling_changes_only_when_selected_cell_is_infeasible(self):
        scenario = WildfireSearchScenario()
        scenario.fire_grid_size = 16
        base_mask = torch.ones(16, 16, dtype=torch.bool)

        selected = scenario._sample_random_cell_from_mask(
            base_mask,
            generator=torch.Generator().manual_seed(1000),
        )
        selected_x, selected_y = selected
        unrelated_x = (selected_x + 1) % 16
        unrelated_y = selected_y
        unrelated_mask = base_mask.clone()
        unrelated_mask[unrelated_y, unrelated_x] = False

        unchanged = scenario._sample_random_cell_from_mask(
            unrelated_mask,
            generator=torch.Generator().manual_seed(1000),
        )
        self.assertEqual(unchanged, selected)

        selected_cell_invalid = base_mask.clone()
        selected_cell_invalid[selected_y, selected_x] = False
        fallback = scenario._sample_random_cell_from_mask(
            selected_cell_invalid,
            generator=torch.Generator().manual_seed(1000),
        )
        self.assertNotEqual(fallback, selected)
        self.assertTrue(bool(selected_cell_invalid[fallback[1], fallback[0]].item()))

    def test_cell_sampling_generator_does_not_advance_global_rng(self):
        scenario = WildfireSearchScenario()
        scenario.fire_grid_size = 16
        mask = torch.ones(16, 16, dtype=torch.bool)
        torch.manual_seed(2468)
        state_before = torch.random.get_rng_state().clone()

        scenario._sample_random_cell_from_mask(
            mask,
            generator=torch.Generator().manual_seed(1000),
        )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))


class ResetRngContextTests(unittest.TestCase):
    @staticmethod
    def _draw(context: _ResetRngContext, stream: str, slot_index: int | None = None):
        return torch.randint(
            0,
            2**31,
            (8,),
            generator=context.generator(stream, slot_index),
        )

    def test_identical_reset_identity_repeats_stream(self):
        first = _ResetRngContext(base_seed=1000, env_index=2, episode_index=4)
        second = _ResetRngContext(base_seed=1000, env_index=2, episode_index=4)

        self.assertTrue(torch.equal(
            self._draw(first, "survivor_positions", 3),
            self._draw(second, "survivor_positions", 3),
        ))

    def test_reset_identity_components_have_distinct_seeds(self):
        contexts = (
            (_ResetRngContext(base_seed=1000, env_index=2, episode_index=4), "survivor_positions", None),
            (_ResetRngContext(base_seed=1001, env_index=2, episode_index=4), "survivor_positions", None),
            (_ResetRngContext(base_seed=1000, env_index=3, episode_index=4), "survivor_positions", None),
            (_ResetRngContext(base_seed=1000, env_index=2, episode_index=5), "survivor_positions", None),
            (_ResetRngContext(base_seed=1000, env_index=2, episode_index=4), "decoy_positions", None),
            (_ResetRngContext(base_seed=1000, env_index=2, episode_index=4), "survivor_positions", 0),
            (_ResetRngContext(base_seed=1000, env_index=2, episode_index=4), "survivor_positions", 1),
        )
        seeds = [context.stream_seed(stream, slot) for context, stream, slot in contexts]

        self.assertEqual(len(seeds), len(set(seeds)))

    def test_cached_generator_advances_its_local_sequence(self):
        context = _ResetRngContext(base_seed=1000, env_index=0, episode_index=0)
        reference = _ResetRngContext(base_seed=1000, env_index=0, episode_index=0)
        generator = context.generator("survivor_positions")

        first = torch.randint(0, 2**31, (8,), generator=generator)
        second = torch.randint(
            0,
            2**31,
            (8,),
            generator=context.generator("survivor_positions"),
        )
        expected = torch.randint(
            0,
            2**31,
            (16,),
            generator=reference.generator("survivor_positions"),
        )

        self.assertIs(generator, context.generator("survivor_positions"))
        self.assertTrue(torch.equal(torch.cat((first, second)), expected))

    def test_slot_streams_do_not_depend_on_request_order(self):
        first = _ResetRngContext(base_seed=1000, env_index=0, episode_index=0)
        second = _ResetRngContext(base_seed=1000, env_index=0, episode_index=0)

        first_slot_zero = self._draw(first, "uav_starts", 0)
        first_slot_one = self._draw(first, "uav_starts", 1)
        second_slot_one = self._draw(second, "uav_starts", 1)
        second_slot_zero = self._draw(second, "uav_starts", 0)

        self.assertTrue(torch.equal(first_slot_zero, second_slot_zero))
        self.assertTrue(torch.equal(first_slot_one, second_slot_one))

    def test_reset_generator_does_not_advance_global_rng(self):
        torch.manual_seed(2468)
        state_before = torch.random.get_rng_state().clone()
        context = _ResetRngContext(base_seed=1000, env_index=0, episode_index=0)

        self._draw(context, "uav_starts")

        self.assertTrue(torch.equal(torch.random.get_rng_state(), state_before))

    def test_partial_reset_order_does_not_change_context_seeds(self):
        first = ScenarioRngInvarianceTests()._env(
            seed=1000,
            n_drones=3,
            n_ground=2,
            num_envs=2,
        )
        second = ScenarioRngInvarianceTests()._env(
            seed=1000,
            n_drones=3,
            n_ground=2,
            num_envs=2,
        )
        first.reset(seed=1000)
        second.reset(seed=1000)

        first.reset_at(index=0, return_observations=False)
        first.reset_at(index=1, return_observations=False)
        second.reset_at(index=1, return_observations=False)
        second.reset_at(index=0, return_observations=False)

        for env_index in (0, 1):
            first_context = first.scenario._reset_rng_context(env_index)
            second_context = second.scenario._reset_rng_context(env_index)
            self.assertEqual(first_context.base_seed, second_context.base_seed)
            self.assertEqual(first_context.episode_index, second_context.episode_index)
            self.assertEqual(
                first_context.stream_seed("survivor_positions"),
                second_context.stream_seed("survivor_positions"),
            )

    def test_episode_index_tracks_partial_and_full_resets(self):
        fixture = ScenarioRngInvarianceTests()
        env = fixture._env(seed=1000, n_drones=3, n_ground=2, num_envs=2)

        env.reset(seed=1000)
        self.assertEqual(env.scenario._reset_rng_context(0).episode_index, 0)
        self.assertEqual(env.scenario._reset_rng_context(1).episode_index, 0)

        env.reset_at(index=1, return_observations=False)
        self.assertEqual(env.scenario._reset_rng_context(0).episode_index, 0)
        self.assertEqual(env.scenario._reset_rng_context(1).episode_index, 1)

        env.reset(return_observations=False)
        self.assertEqual(env.scenario._reset_rng_context(0).episode_index, 1)
        self.assertEqual(env.scenario._reset_rng_context(1).episode_index, 2)

        env.reset(seed=1000, return_observations=False)
        self.assertEqual(env.scenario._reset_rng_context(0).episode_index, 0)
        self.assertEqual(env.scenario._reset_rng_context(1).episode_index, 0)

    def test_reset_identity_does_not_depend_on_global_rng_consumption(self):
        fixture = ScenarioRngInvarianceTests()
        env = fixture._env(seed=1000, n_drones=3, n_ground=2, num_envs=1)
        scenario = env.scenario
        torch.manual_seed(1000)

        scenario._notify_explicit_reset_seed(1000)
        first = scenario._begin_reset_rng(None)[0]
        second = scenario._begin_reset_rng(None)[0]
        scenario._notify_explicit_reset_seed(1000)
        reseeded = scenario._begin_reset_rng(None)[0]

        self.assertEqual(first.episode_index, 0)
        self.assertEqual(second.episode_index, 1)
        self.assertEqual(reseeded.episode_index, 0)

    def test_environment_hands_explicit_seed_to_scenario(self):
        fixture = ScenarioRngInvarianceTests()
        env = fixture._env(seed=1000, n_drones=3, n_ground=2, num_envs=1)

        env.reset(return_observations=False)
        self.assertEqual(env.scenario._reset_rng_context(0).episode_index, 1)

        env.reset(seed=1000, return_observations=False)
        context = env.scenario._reset_rng_context(0)
        self.assertEqual(context.base_seed, 1000)
        self.assertEqual(context.episode_index, 0)


if __name__ == "__main__":
    unittest.main()

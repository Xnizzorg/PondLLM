import unittest

from pondllm.domain import Action, ActionKind, Genes
from pondllm.policies import HeuristicPolicy
from pondllm.world import World, WorldConfig


def small_config(**overrides):
    values = {
        "width": 7,
        "height": 7,
        "founders": 1,
        "initial_energy": 10,
        "initial_food": 0,
        "food_energy": 4,
        "food_regrowth": 0,
        "max_food": 20,
        "metabolism": 1,
        "reproduction_cost": 8,
        "child_energy": 4,
        "max_population": 8,
        "memory_limit": 4,
        "perception_radius": 2,
    }
    values.update(overrides)
    return WorldConfig(**values)


class WorldTests(unittest.TestCase):
    def test_observation_is_local(self) -> None:
        world = World(small_config(), seed=1, initialize=False)
        actor = world.spawn_founder((1, 1), "a")
        nearby = world.spawn_founder((2, 1), "b")
        world.spawn_founder((6, 6), "hidden")
        world.add_food((1, 2))
        world.add_food((6, 6))

        observation = world.observe(actor)
        self.assertEqual([item["id"] for item in observation.visible_agents], [nearby.organism_id])
        self.assertEqual(observation.visible_food, ((1, 2, 1),))

    def test_forage_changes_energy_and_consumes_food(self) -> None:
        world = World(small_config(), seed=1, initialize=False)
        actor = world.spawn_founder((1, 1), "a", energy=5)
        world.add_food((1, 1))
        valid, _ = world.apply_action(actor, Action(ActionKind.FORAGE))
        self.assertTrue(valid)
        self.assertEqual(actor.energy, 9)
        self.assertNotIn((1, 1), world.food)

    def test_reproduction_records_ancestry(self) -> None:
        world = World(small_config(), seed=2, initialize=False)
        actor = world.spawn_founder(
            (2, 2),
            "a",
            energy=20,
            genes=Genes(reproduction_threshold=12),
        )
        valid, _ = world.apply_action(
            actor,
            Action(ActionKind.REPRODUCE, target_position=(3, 2)),
        )
        self.assertTrue(valid)
        children = [item for item in world.living() if item.parent_id == actor.organism_id]
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].generation, 1)
        self.assertEqual(children[0].lineage_id, actor.lineage_id)
        self.assertEqual(actor.energy, 12)

    def test_fixed_seed_is_reproducible(self) -> None:
        config = small_config(founders=4, initial_food=12, food_regrowth=2)
        first = World(config, seed=42)
        second = World(config, seed=42)
        first_summary = first.run(HeuristicPolicy(99), 30)
        second_summary = second.run(HeuristicPolicy(99), 30)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            [event.to_dict() for event in first.events],
            [event.to_dict() for event in second.events],
        )


if __name__ == "__main__":
    unittest.main()


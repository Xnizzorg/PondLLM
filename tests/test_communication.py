import re
import tempfile
import unittest
from pathlib import Path

from pondllm.communication import (
    COMMUNICATION_CONDITIONS,
    create_communication_world,
    run_communication_experiment,
)
from pondllm.domain import Action, ActionKind, Observation
from pondllm.world import World, WorldConfig


class ScriptedCommunicationPolicy:
    def choose(self, observation: Observation) -> Action:
        if observation.current_food > 0:
            return Action(ActionKind.FORAGE)
        coordinate = self._remembered_food(observation)
        if coordinate is not None:
            best_distance = min(
                _manhattan(position, coordinate)
                for position in observation.open_neighbors
            )
            target = min(
                position
                for position in observation.open_neighbors
                if _manhattan(position, coordinate) == best_distance
            )
            return Action(ActionKind.MOVE, target_position=target)

        useful_food = [
            food
            for food in observation.visible_food
            if any(
                _manhattan(agent["position"], food[:2])
                > observation.perception_radius
                for agent in observation.visible_agents
            )
        ]
        if useful_food:
            target = useful_food[0][:2]
            return Action(ActionKind.SIGNAL, message=f"food at [{target[0]},{target[1]}]")

        return Action(ActionKind.REST)

    @staticmethod
    def _remembered_food(observation: Observation) -> tuple[int, int] | None:
        for item in reversed(observation.memory):
            match = re.search(r"food at \[(-?\d+),(-?\d+)\]", item)
            if match:
                return int(match.group(1)), int(match.group(2))
        return None


class CommunicationWorldTests(unittest.TestCase):
    def test_signal_interventions_are_logged_and_deterministic(self) -> None:
        normal, normal_recipient, normal_sender = _signal_world("normal", 0)
        valid, _ = normal.apply_action(
            normal_sender,
            Action(ActionKind.SIGNAL, message="food at [4,2]"),
        )
        self.assertTrue(valid)
        self.assertIn("food at [4,2]", normal_recipient.memory[-1])
        self.assertEqual(normal.events[-1].data["delivery"], "normal")

        blocked, blocked_recipient, blocked_sender = _signal_world("blocked", 0)
        blocked.apply_action(
            blocked_sender,
            Action(ActionKind.SIGNAL, message="food at [4,2]"),
        )
        self.assertEqual(blocked_recipient.memory, [])
        self.assertIsNone(blocked.events[-1].data["delivered_message"])

        corrupted, corrupted_recipient, corrupted_sender = _signal_world("corrupted", 0)
        corrupted.apply_action(
            corrupted_sender,
            Action(ActionKind.SIGNAL, message="food at [4,2]"),
        )
        self.assertIn("food at [0,5]", corrupted_recipient.memory[-1])
        self.assertNotIn("food at [4,2]", corrupted_recipient.memory[-1])

        costly, costly_recipient, costly_sender = _signal_world("normal", 1)
        costly.apply_action(
            costly_sender,
            Action(ActionKind.SIGNAL, message="food at [4,2]"),
        )
        self.assertEqual(costly_sender.energy, 9)
        self.assertTrue(costly_recipient.memory)
        self.assertEqual(costly.events[-1].data["cost"], 1)

    def test_paired_scenes_differ_only_by_channel(self) -> None:
        scenes = []
        sender_observations = []
        recipient_observations = []
        for condition in COMMUNICATION_CONDITIONS:
            world, scene = create_communication_world(102, condition)
            scenes.append(scene)
            sender_observations.append(world.observe(world.organisms[scene.sender_id]).to_dict())
            recipient_observations.append(
                world.observe(world.organisms[scene.recipient_id]).to_dict()
            )
        self.assertTrue(all(scene == scenes[0] for scene in scenes))
        self.assertTrue(
            all(observation == sender_observations[0] for observation in sender_observations)
        )
        self.assertTrue(
            all(
                observation == recipient_observations[0]
                for observation in recipient_observations
            )
        )
        self.assertTrue(sender_observations[0]["visible_food"])
        self.assertEqual(recipient_observations[0]["visible_food"], [])

    def test_clean_profile_removes_training_only_context_cues(self) -> None:
        matched, matched_scene = create_communication_world(102, "normal", profile="matched")
        clean, clean_scene = create_communication_world(102, "normal", profile="clean")
        matched_sender = matched.observe(matched.organisms[matched_scene.sender_id])
        clean_sender = clean.observe(clean.organisms[clean_scene.sender_id])
        self.assertGreater(matched_sender.tick, 0)
        self.assertTrue(matched_sender.memory)
        self.assertEqual(clean_sender.tick, 0)
        self.assertEqual(clean_sender.memory, ())
        self.assertEqual(matched_sender.position, clean_sender.position)
        self.assertEqual(matched_sender.visible_food, clean_sender.visible_food)
        self.assertEqual(
            matched_sender.visible_agents[0]["position"],
            clean_sender.visible_agents[0]["position"],
        )
        self.assertTrue(matched_sender.organism_id.startswith("organism-s"))
        self.assertTrue(matched_sender.visible_agents[0]["id"].startswith("organism-r"))
        self.assertTrue(clean_sender.organism_id.startswith("organism-000"))
        self.assertEqual(clean_sender.lineage_id, "lineage-00001")
        self.assertEqual(clean_sender.visible_agents[0]["lineage"], "lineage-00002")

    def test_v4_profile_is_neutral_and_geometrically_diverse(self) -> None:
        scenes = [
            create_communication_world(seed, "normal", profile="v4")[1]
            for seed in range(200, 212)
        ]
        layouts = {
            (
                scene.sender_position,
                scene.recipient_position,
                scene.target_food,
            )
            for scene in scenes
        }
        self.assertGreater(len(layouts), 8)
        for seed, scene in zip(range(200, 212), scenes, strict=True):
            world, recreated = create_communication_world(
                seed,
                "normal",
                profile="v4",
            )
            self.assertEqual(scene, recreated)
            sender = world.observe(world.organisms[scene.sender_id])
            self.assertNotIn("-s", sender.organism_id)
            self.assertNotIn("-r", sender.organism_id)
            self.assertLessEqual(sender.age, sender.tick)
            self.assertTrue(sender.visible_food)
            self.assertGreater(
                _manhattan(
                    sender.visible_agents[0]["position"],
                    sender.visible_food[0][:2],
                ),
                sender.perception_radius,
            )

    def test_scripted_policy_proves_normal_channel_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_communication_experiment(
                policy=ScriptedCommunicationPolicy(),
                output_dir=Path(temporary),
                seeds=[100, 101, 102, 103],
                steps=7,
            )
            normal = summary["per_condition"]["normal"]
            blocked = summary["per_condition"]["blocked"]
            corrupted = summary["per_condition"]["corrupted"]
            costly = summary["per_condition"]["costly"]
            self.assertEqual(normal["recipient_informed_rate"], 1.0)
            self.assertEqual(blocked["recipient_informed_rate"], 0.0)
            self.assertEqual(normal["recipient_reached_food_rate"], 1.0)
            self.assertEqual(blocked["recipient_reached_food_rate"], 0.0)
            self.assertLess(
                normal["mean_recipient_final_distance"],
                corrupted["mean_recipient_final_distance"],
            )
            self.assertGreater(costly["total_signal_energy_spent"], 0)
            self.assertGreater(summary["paired_effects"]["normal_distance_advantage"], 0)

    def test_scripted_policy_proves_diverse_v4_channel_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = run_communication_experiment(
                policy=ScriptedCommunicationPolicy(),
                output_dir=Path(temporary),
                seeds=range(200, 212),
                steps=11,
                profile="v4",
            )
            normal = summary["per_condition"]["normal"]
            blocked = summary["per_condition"]["blocked"]
            self.assertEqual(normal["sender_signalled_rate"], 1.0)
            self.assertEqual(normal["recipient_foraged_rate"], 1.0)
            self.assertEqual(blocked["recipient_foraged_rate"], 0.0)
            self.assertEqual(
                summary["paired_effects"]["normal_minus_blocked_forage_rate"],
                1.0,
            )

    def test_v41_profile_adds_rich_context_and_preserves_channel_effect(self) -> None:
        world, scene = create_communication_world(
            400,
            "normal",
            profile="v41",
        )
        sender = world.observe(world.organisms[scene.sender_id])
        recipient = world.observe(world.organisms[scene.recipient_id])
        self.assertEqual(len(sender.visible_agents), 3)
        self.assertEqual(len(sender.visible_food), 2)
        self.assertEqual(recipient.visible_food, ())
        self.assertEqual(len(scene.distractor_ids), 2)

        with tempfile.TemporaryDirectory() as temporary:
            summary = run_communication_experiment(
                policy=ScriptedCommunicationPolicy(),
                output_dir=Path(temporary),
                seeds=range(400, 408),
                steps=11,
                profile="v41",
            )
            normal = summary["per_condition"]["normal"]
            blocked = summary["per_condition"]["blocked"]
            self.assertEqual(normal["sender_signalled_rate"], 1.0)
            self.assertEqual(normal["recipient_foraged_rate"], 1.0)
            self.assertEqual(blocked["recipient_foraged_rate"], 0.0)
            self.assertEqual(
                summary["paired_effects"]["normal_minus_blocked_forage_rate"],
                1.0,
            )


def _signal_world(delivery: str, cost: int) -> tuple[World, object, object]:
    config = WorldConfig(
        width=7,
        height=7,
        founders=2,
        initial_food=0,
        food_regrowth=0,
        max_population=2,
        signal_delivery=delivery,
        signal_cost=cost,
    )
    world = World(config, seed=1, initialize=False)
    sender = world.spawn_founder((2, 2), "sender", energy=10)
    recipient = world.spawn_founder((2, 4), "recipient", energy=10)
    return world, recipient, sender


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


if __name__ == "__main__":
    unittest.main()

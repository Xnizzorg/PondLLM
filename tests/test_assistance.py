import tempfile
import unittest
from pathlib import Path

from pondllm.assistance import create_rescue_world, run_rescue_experiment
from pondllm.domain import Action, ActionKind, Observation


class ScriptedRescuePolicy:
    def choose(self, observation: Observation) -> Action:
        if observation.current_food:
            return Action(ActionKind.FORAGE)
        adjacent_needy = [
            agent
            for agent in observation.visible_agents
            if agent["distance"] == 1
            and agent["energy"] == 1
            and not any(
                tuple(food[:2]) == tuple(agent["position"])
                for food in observation.visible_food
            )
        ]
        if (
            adjacent_needy
            and observation.energy
            >= int(observation.drives["share_threshold"])
        ):
            return Action(
                ActionKind.SHARE,
                target_id=adjacent_needy[0]["id"],
                amount=1,
            )
        if observation.visible_food:
            target = observation.visible_food[0][:2]
            distance = _manhattan(observation.position, target)
            if observation.energy <= distance:
                return Action(ActionKind.REST)
            move = min(
                observation.open_neighbors,
                key=lambda position: (_manhattan(position, target), position),
            )
            return Action(ActionKind.MOVE, target_position=move)
        return Action(ActionKind.REST)


class AssistanceTests(unittest.TestCase):
    def test_rescue_interventions_are_paired_and_logged(self) -> None:
        normal, scene = create_rescue_world(900, "normal")
        blocked, blocked_scene = create_rescue_world(900, "blocked")
        self.assertEqual(scene, blocked_scene)
        self.assertEqual(
            normal.observe(normal.organisms[scene.child_id]).energy,
            1,
        )
        self.assertEqual(
            blocked.observe(blocked.organisms[scene.child_id]).visible_food[0][:2],
            scene.target_food,
        )

        with tempfile.TemporaryDirectory() as directory:
            summary = run_rescue_experiment(
                policy=ScriptedRescuePolicy(),
                output_dir=Path(directory),
                seeds=range(900, 904),
                steps=5,
            )
            normal_summary = summary["per_condition"]["normal"]
            blocked_summary = summary["per_condition"]["blocked"]
            unsafe_summary = summary["per_condition"]["unsafe"]
            self.assertEqual(normal_summary["donor_share_rate"], 1.0)
            self.assertEqual(normal_summary["child_forage_rate"], 1.0)
            self.assertEqual(normal_summary["total_repeated_shares"], 0)
            self.assertEqual(normal_summary["total_child_boundary_moves"], 0)
            self.assertEqual(blocked_summary["child_wait_rate"], 1.0)
            self.assertEqual(blocked_summary["child_survival_rate"], 1.0)
            self.assertEqual(blocked_summary["child_forage_rate"], 0.0)
            self.assertEqual(unsafe_summary["total_unsafe_shares"], 0)
            self.assertEqual(
                summary["paired_effects"]["normal_minus_blocked_forage_rate"],
                1.0,
            )


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


if __name__ == "__main__":
    unittest.main()

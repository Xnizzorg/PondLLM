import unittest

from pondllm.domain import Action, ActionKind, InvalidAction
from pondllm.prompting import action_is_observation_legal, parse_action_text


class ActionTests(unittest.TestCase):
    def test_action_round_trip(self) -> None:
        action = Action.from_payload({"action": "move", "target": [2, 3]})
        self.assertEqual(action.kind, ActionKind.MOVE)
        self.assertEqual(action.target_position, (2, 3))
        self.assertEqual(action.to_dict(), {"action": "move", "target": [2, 3]})

    def test_rejects_invalid_share_amount(self) -> None:
        with self.assertRaises(InvalidAction):
            Action.from_payload({"action": "share", "target": "other", "amount": 0})

    def test_parser_handles_fenced_json(self) -> None:
        action = parse_action_text('```json\n{"action":"forage"}\n```')
        self.assertEqual(action, Action(ActionKind.FORAGE))

    def test_parser_ignores_leading_text(self) -> None:
        action = parse_action_text('Action follows: {"action":"rest"}')
        self.assertEqual(action, Action(ActionKind.REST))

    def test_observation_level_legality(self) -> None:
        observation = {
            "self": {"energy": 12, "drives": {"reproduction_threshold": 10}},
            "current_food": 0,
            "open_neighbors": [[2, 1]],
            "visible_agents": [{"id": "other", "distance": 1, "energy": 3}],
        }
        self.assertTrue(
            action_is_observation_legal(
                Action(ActionKind.MOVE, target_position=(2, 1)), observation
            )[0]
        )
        self.assertFalse(action_is_observation_legal(Action(ActionKind.FORAGE), observation)[0])


if __name__ == "__main__":
    unittest.main()

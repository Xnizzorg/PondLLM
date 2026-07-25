import json
import tempfile
import unittest
from pathlib import Path

from pondllm.dataset import (
    generate_balanced_sft_dataset,
    generate_communication_sft_dataset,
    generate_sft_dataset,
    stratify_sft_dataset,
)
from pondllm.domain import Action, ActionKind
from pondllm.prompting import training_record
from pondllm.world import WorldConfig


class DatasetTests(unittest.TestCase):
    def test_generates_conversational_prompt_completion_jsonl(self) -> None:
        config = WorldConfig(
            width=7,
            height=7,
            founders=3,
            initial_food=12,
            food_regrowth=1,
            max_food=20,
            max_population=8,
            perception_radius=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            summary = generate_sft_dataset(path, config, episodes=2, steps_per_episode=8, seed=3)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), summary["records"])
            self.assertGreater(len(lines), 0)
            record = json.loads(lines[0])
            self.assertEqual(record["prompt"][0]["role"], "system")
            self.assertEqual(record["completion"][0]["role"], "assistant")
            self.assertTrue(record["completion"][0]["content"].startswith("{"))
            self.assertTrue(path.with_suffix(".summary.json").is_file())

    def test_balances_every_action_deterministically(self) -> None:
        config = WorldConfig(
            width=24,
            height=24,
            founders=12,
            initial_energy=10,
            initial_food=55,
            food_energy=4,
            food_regrowth=3,
            max_food=80,
            max_population=48,
            perception_radius=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.jsonl"
            first = Path(directory) / "balanced-first.jsonl"
            second = Path(directory) / "balanced-second.jsonl"
            generate_sft_dataset(base, config, episodes=1, steps_per_episode=20, seed=1000)
            first_summary = generate_balanced_sft_dataset(
                first,
                base,
                config,
                episodes=32,
                steps_per_episode=60,
                seed=2000,
                minimum_per_action=2,
            )
            second_summary = generate_balanced_sft_dataset(
                second,
                base,
                config,
                episodes=32,
                steps_per_episode=60,
                seed=2000,
                minimum_per_action=2,
            )
            self.assertTrue(
                all(count >= 2 for count in first_summary["action_counts"].values())
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_summary["action_counts"],
                second_summary["action_counts"],
            )

    def test_stratifies_equal_action_coverage(self) -> None:
        observation = {
            "self": {"energy": 20, "drives": {"reproduction_threshold": 16}},
            "current_food": 1,
            "open_neighbors": [[1, 0]],
            "visible_agents": [{"id": "other", "distance": 1, "energy": 2}],
        }
        actions = {
            ActionKind.MOVE: Action(ActionKind.MOVE, target_position=(1, 0)),
            ActionKind.FORAGE: Action(ActionKind.FORAGE),
            ActionKind.SHARE: Action(ActionKind.SHARE, target_id="other", amount=1),
            ActionKind.SIGNAL: Action(ActionKind.SIGNAL, message="here"),
            ActionKind.REST: Action(ActionKind.REST),
            ActionKind.REPRODUCE: Action(ActionKind.REPRODUCE, target_position=(1, 0)),
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "stratified.jsonl"
            records = [
                training_record(observation, action)
                for action in actions.values()
                for _ in range(2)
            ]
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = stratify_sft_dataset(source, output, records_per_action=1, seed=11)
            self.assertEqual(summary["records"], len(ActionKind))
            self.assertEqual(set(summary["action_counts"].values()), {1})

    def test_communication_dataset_is_paired_and_deterministic(self) -> None:
        config = WorldConfig(
            width=12,
            height=12,
            founders=2,
            initial_food=0,
            food_regrowth=0,
            max_food=10,
            max_population=4,
            perception_radius=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "communication-first.jsonl"
            second = Path(directory) / "communication-second.jsonl"
            first_summary = generate_communication_sft_dataset(
                first, config, scenes=8, seed=31
            )
            second_summary = generate_communication_sft_dataset(
                second, config, scenes=8, seed=31
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_summary["records"], 32)
            self.assertEqual(
                first_summary["communication_case_counts"],
                {
                    "recipient_control": 8,
                    "recipient_informed": 8,
                    "sender_redundant": 8,
                    "sender_useful": 8,
                },
            )
            records = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            by_pair = {}
            for record in records:
                by_pair.setdefault(record["pair_id"], []).append(record)
            self.assertTrue(all(len(pair) == 2 for pair in by_pair.values()))


if __name__ == "__main__":
    unittest.main()

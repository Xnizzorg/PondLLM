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
from pondllm.curriculum_v4 import generate_v4_sft_dataset
from pondllm.curriculum_v41 import generate_v41_sft_dataset
from pondllm.curriculum_v42 import audit_v42_datasets, generate_v42_sft_dataset
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

    def test_v4_dataset_uses_neutral_simulator_observations(self) -> None:
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
            first = Path(directory) / "v4-first.jsonl"
            second = Path(directory) / "v4-second.jsonl"
            first_summary = generate_v4_sft_dataset(
                first,
                config,
                scenes=8,
                survival_scenes=2,
                seed=53,
            )
            second_summary = generate_v4_sft_dataset(
                second,
                config,
                scenes=8,
                survival_scenes=2,
                seed=53,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_summary["sha256"], second_summary["sha256"])
            records = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            communication = [
                record for record in records if "communication_case" in record
            ]
            observations = [
                json.loads(record["prompt"][1]["content"])
                for record in communication
            ]
            self.assertTrue(observations)
            self.assertFalse(
                any(
                    "-s" in observation["self"]["id"]
                    or "-r" in observation["self"]["id"]
                    or "-s" in observation["self"]["lineage"]
                    or "-r" in observation["self"]["lineage"]
                    for observation in observations
                )
            )
            self.assertFalse(
                any(
                    "last forage was" in item or "survived tick" in item
                    for observation in observations
                    for item in observation["memory"]
                )
            )
            self.assertTrue(
                all(
                    observation["self"]["age"] <= observation["tick"]
                    for observation in observations
                )
            )

    def test_v41_dataset_is_rich_reachable_and_deterministic(self) -> None:
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
            first = Path(directory) / "v41-first.jsonl"
            second = Path(directory) / "v41-second.jsonl"
            first_summary = generate_v41_sft_dataset(
                first,
                config,
                rich_scenes=6,
                reachability_scenes=6,
                seed=61,
            )
            second_summary = generate_v41_sft_dataset(
                second,
                config,
                rich_scenes=6,
                reachability_scenes=6,
                seed=61,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_summary["sha256"], second_summary["sha256"])
            self.assertEqual(first_summary["duplicate_records_skipped"], 0)
            records = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            rich_useful = [
                record
                for record in records
                if record.get("communication_case") == "rich_sender_useful"
            ]
            self.assertEqual(len(rich_useful), 6)
            for record in rich_useful:
                observation = json.loads(record["prompt"][1]["content"])
                self.assertEqual(len(observation["visible_agents"]), 3)
                self.assertEqual(len(observation["visible_food"]), 2)
            unreachable = [
                record
                for record in records
                if record.get("survival_case") == "food_unreachable"
            ]
            self.assertEqual(len(unreachable), 6)
            self.assertTrue(
                all(
                    json.loads(record["completion"][0]["content"])["action"]
                    == "rest"
                    and record["reachable_before_exhaustion"] is False
                    for record in unreachable
                )
            )

    def test_v42_dataset_hardens_negatives_and_energy_trajectories(self) -> None:
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
            first = Path(directory) / "v42-first.jsonl"
            second = Path(directory) / "v42-second.jsonl"
            held_out = Path(directory) / "v42-held-out.jsonl"
            first_summary = generate_v42_sft_dataset(
                first,
                config,
                paired_scenes=2,
                redundant_scenes=4,
                trajectory_scenes=2,
                seed=81,
            )
            second_summary = generate_v42_sft_dataset(
                second,
                config,
                paired_scenes=2,
                redundant_scenes=4,
                trajectory_scenes=2,
                seed=81,
            )
            generate_v42_sft_dataset(
                held_out,
                config,
                paired_scenes=2,
                redundant_scenes=2,
                trajectory_scenes=2,
                seed=91,
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_summary["sha256"], second_summary["sha256"])
            self.assertEqual(first_summary["records"], 38)
            self.assertEqual(
                first_summary["communication_case_counts"],
                {"rich_sender_redundant": 6, "rich_sender_useful": 2},
            )
            self.assertEqual(
                set(first_summary["energy_case_counts"].values()),
                {2},
            )
            records = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
            boundary = [
                record
                for record in records
                if record.get("energy_case") in {
                    "boundary_one_wait",
                    "boundary_two_wait",
                }
            ]
            self.assertTrue(boundary)
            for record in boundary:
                observation = json.loads(record["prompt"][1]["content"])
                self.assertEqual(
                    observation["self"]["energy"],
                    record["target_distance"],
                )
                self.assertEqual(
                    json.loads(record["completion"][0]["content"])["action"],
                    "rest",
                )
            audit = audit_v42_datasets(first, [held_out])
            self.assertTrue(audit["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()

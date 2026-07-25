import json
import tempfile
import unittest
from pathlib import Path

from pondllm.domain import Action, ActionKind
from pondllm.evaluation import evaluate_communication_policy, evaluate_policy


class RestPolicy:
    last_raw_output = '{"action":"rest"}'

    def choose(self, observation):
        return Action(ActionKind.REST)


class EvaluationTests(unittest.TestCase):
    def test_evaluates_syntax_and_legality(self) -> None:
        observation = {
            "self": {"energy": 4, "drives": {"reproduction_threshold": 16}},
            "current_food": 0,
            "open_neighbors": [[1, 0]],
            "visible_agents": [],
        }
        record = {
            "prompt": [
                {"role": "system", "content": "test"},
                {"role": "user", "content": json.dumps(observation)},
            ],
            "completion": [{"role": "assistant", "content": '{"action":"rest"}'}],
        }
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            predictions = Path(directory) / "predictions.jsonl"
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary = evaluate_policy(RestPolicy(), dataset, predictions, max_records=1)
            self.assertEqual(summary["syntax_valid_rate"], 1.0)
            self.assertEqual(summary["legal_rate"], 1.0)
            self.assertEqual(summary["macro_action_kind_accuracy"], 1.0)
            self.assertEqual(summary["per_action"]["rest"]["recall"], 1.0)
            self.assertTrue(predictions.is_file())

    def test_communication_metrics_keep_sender_failures_separate(self) -> None:
        observation = {
            "self": {"energy": 8, "drives": {"reproduction_threshold": 16}},
            "current_food": 0,
            "open_neighbors": [[1, 0]],
            "visible_agents": [],
        }
        records = []
        for case in ("sender_useful", "sender_redundant"):
            records.append(
                {
                    "prompt": [
                        {"role": "system", "content": "test"},
                        {"role": "user", "content": json.dumps(observation)},
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": json.dumps(Action(ActionKind.REST).to_dict()),
                        }
                    ],
                    "communication_case": case,
                    "pair_id": "pair-1",
                    "target_food": [2, 0],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "communication.jsonl"
            predictions = Path(directory) / "predictions.jsonl"
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = evaluate_communication_policy(RestPolicy(), dataset, predictions)
            self.assertEqual(summary["syntax_valid_rate"], 1.0)
            self.assertEqual(summary["useful_signal_rate"], 0.0)
            self.assertEqual(summary["redundant_signal_rate"], 0.0)
            self.assertEqual(summary["paired_exact_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

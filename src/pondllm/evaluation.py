from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from .domain import Action, ActionKind, InvalidAction
from .prompting import action_is_observation_legal, strict_json_action


def evaluate_policy(
    policy: Any,
    dataset_path: str | Path,
    output_path: str | Path,
    max_records: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    if max_records < 1:
        raise ValueError("max_records must be positive")
    records = _read_records(dataset_path)
    Random(seed).shuffle(records)
    records = records[:max_records]

    syntax_valid = 0
    strict_json = 0
    legal = 0
    kind_matches = 0
    exact_matches = 0
    action_counts: Counter[str] = Counter()
    expected_action_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    survival_totals: Counter[str] = Counter()
    survival_exact: Counter[str] = Counter()
    survival_shares: Counter[str] = Counter()
    predictions = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        observation = _observation_from_record(record)
        expected = json.loads(record["completion"][0]["content"])
        expected_action = Action.from_payload(expected)
        survival_case = record.get("survival_case")
        if isinstance(survival_case, str):
            survival_totals[survival_case] += 1
        expected_action_counts[expected_action.kind.value] += 1
        prediction: Action | None = None
        error: str | None = None
        legal_reason: str | None = None
        try:
            prediction = policy.choose(observation)
            syntax_valid += 1
            strict_json += int(
                strict_json_action(getattr(policy, "last_raw_output", None)) is not None
            )
            action_counts[prediction.kind.value] += 1
            kind_matches += int(prediction.kind is expected_action.kind)
            exact_matches += int(prediction.to_dict() == expected_action.to_dict())
            if isinstance(survival_case, str):
                survival_exact[survival_case] += int(
                    prediction.to_dict() == expected_action.to_dict()
                )
                survival_shares[survival_case] += int(
                    prediction.kind is ActionKind.SHARE
                )
            confusion[f"{expected_action.kind.value}->{prediction.kind.value}"] += 1
            is_legal, legal_reason = action_is_observation_legal(prediction, observation)
            legal += int(is_legal)
        except (InvalidAction, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        predictions.append(
            {
                "index": index,
                "observation": observation,
                "expected": expected,
                "predicted": prediction.to_dict() if prediction else None,
                "raw_output": getattr(policy, "last_raw_output", None),
                "strict_json": (
                    strict_json_action(getattr(policy, "last_raw_output", None))
                    is not None
                ),
                "legal_reason": legal_reason,
                "error": error,
            }
        )
    elapsed = time.perf_counter() - started

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    count = len(records)
    per_action = {}
    for kind, expected_count in sorted(expected_action_counts.items()):
        matches = confusion[f"{kind}->{kind}"]
        per_action[kind] = {
            "expected": expected_count,
            "matches": matches,
            "recall": round(matches / expected_count, 4) if expected_count else 0.0,
        }
    macro_action_kind_accuracy = (
        round(
            sum(metrics["recall"] for metrics in per_action.values()) / len(per_action),
            4,
        )
        if per_action
        else 0.0
    )
    return {
        "dataset": str(Path(dataset_path).resolve()),
        "predictions": str(destination.resolve()),
        "records": count,
        "syntax_valid": syntax_valid,
        "syntax_valid_rate": round(syntax_valid / count, 4) if count else 0.0,
        "strict_json": strict_json,
        "strict_json_rate": round(strict_json / count, 4) if count else 0.0,
        "legal": legal,
        "legal_rate": round(legal / count, 4) if count else 0.0,
        "action_kind_matches": kind_matches,
        "action_kind_accuracy": round(kind_matches / count, 4) if count else 0.0,
        "exact_matches": exact_matches,
        "exact_accuracy": round(exact_matches / count, 4) if count else 0.0,
        "macro_action_kind_accuracy": macro_action_kind_accuracy,
        "per_action": per_action,
        "expected_action_counts": dict(sorted(expected_action_counts.items())),
        "predicted_action_counts": dict(sorted(action_counts.items())),
        "action_confusion": dict(sorted(confusion.items())),
        "per_survival_case": {
            case: {
                "records": total,
                "exact": survival_exact[case],
                "exact_accuracy": round(survival_exact[case] / total, 4),
                "share_rate": round(survival_shares[case] / total, 4),
            }
            for case, total in sorted(survival_totals.items())
        },
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(count / elapsed, 3) if elapsed else 0.0,
    }


def evaluate_communication_policy(
    policy: Any,
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate causal communication cases without averaging away failure modes."""

    records = _read_records(dataset_path)
    case_totals: Counter[str] = Counter()
    case_exact: Counter[str] = Counter()
    syntax_valid = 0
    strict_json = 0
    legal = 0
    useful_signals = 0
    redundant_signals = 0
    payload_correct = 0
    predictions: list[dict[str, Any]] = []
    pair_results: dict[str, list[bool]] = {}
    started = time.perf_counter()

    for index, record in enumerate(records):
        case = record.get("communication_case")
        pair_id = record.get("pair_id")
        target_food = tuple(record.get("target_food", ()))
        if not isinstance(case, str) or not isinstance(pair_id, str) or len(target_food) != 2:
            raise ValueError(f"record {index} lacks communication metadata")
        observation = _observation_from_record(record)
        expected = Action.from_payload(json.loads(record["completion"][0]["content"]))
        prediction: Action | None = None
        error: str | None = None
        legal_reason: str | None = None
        exact = False
        case_totals[case] += 1
        try:
            prediction = policy.choose(observation)
            syntax_valid += 1
            strict_json += int(
                strict_json_action(getattr(policy, "last_raw_output", None)) is not None
            )
            is_legal, legal_reason = action_is_observation_legal(prediction, observation)
            legal += int(is_legal)
            exact = prediction.to_dict() == expected.to_dict()
            case_exact[case] += int(exact)
            if case in {"sender_useful", "rich_sender_useful"}:
                useful_signals += int(prediction.kind is ActionKind.SIGNAL)
                payload_correct += int(
                    prediction.kind is ActionKind.SIGNAL
                    and _food_coordinate(prediction.message) == target_food
                )
            elif case in {"sender_redundant", "rich_sender_redundant"}:
                redundant_signals += int(prediction.kind is ActionKind.SIGNAL)
        except (InvalidAction, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        pair_results.setdefault(pair_id, []).append(exact)
        predictions.append(
            {
                "index": index,
                "communication_case": case,
                "pair_id": pair_id,
                "target_food": list(target_food),
                "observation": observation,
                "expected": expected.to_dict(),
                "predicted": prediction.to_dict() if prediction else None,
                "raw_output": getattr(policy, "last_raw_output", None),
                "strict_json": (
                    strict_json_action(getattr(policy, "last_raw_output", None))
                    is not None
                ),
                "legal_reason": legal_reason,
                "error": error,
            }
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for prediction_record in predictions:
            handle.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    count = len(records)
    useful_count = (
        case_totals["sender_useful"] + case_totals["rich_sender_useful"]
    )
    redundant_count = (
        case_totals["sender_redundant"] + case_totals["rich_sender_redundant"]
    )
    complete_pairs = [results for results in pair_results.values() if len(results) == 2]
    return {
        "dataset": str(Path(dataset_path).resolve()),
        "predictions": str(destination.resolve()),
        "records": count,
        "syntax_valid_rate": round(syntax_valid / count, 4),
        "strict_json_rate": round(strict_json / count, 4),
        "legal_rate": round(legal / count, 4),
        "exact_accuracy": round(sum(case_exact.values()) / count, 4),
        "per_case": {
            case_name: {
                "records": total,
                "exact": case_exact[case_name],
                "exact_accuracy": round(case_exact[case_name] / total, 4),
            }
            for case_name, total in sorted(case_totals.items())
        },
        "useful_signal_rate": (
            round(useful_signals / useful_count, 4) if useful_count else 0.0
        ),
        "redundant_signal_rate": (
            round(redundant_signals / redundant_count, 4)
            if redundant_count
            else 0.0
        ),
        "payload_coordinate_accuracy": (
            round(payload_correct / useful_count, 4) if useful_count else 0.0
        ),
        "paired_exact_accuracy": (
            round(sum(all(results) for results in complete_pairs) / len(complete_pairs), 4)
            if complete_pairs
            else 0.0
        ),
        "complete_pairs": len(complete_pairs),
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(count / elapsed, 3) if elapsed else 0.0,
    }


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
    if not records:
        raise ValueError("evaluation dataset is empty")
    return records


def _observation_from_record(record: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(record.get("prompt", [])):
        if message.get("role") == "user":
            return json.loads(message["content"])
    raise ValueError("record has no user observation")


def _food_coordinate(message: str | None) -> tuple[int, int] | None:
    if message is None or not message.startswith("food at [") or not message.endswith("]"):
        return None
    values = message[len("food at [") : -1].split(",")
    if len(values) != 2:
        return None
    try:
        return int(values[0].strip()), int(values[1].strip())
    except ValueError:
        return None

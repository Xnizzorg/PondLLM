from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from .domain import Action, ActionKind, Position
from .policies import DemonstrationPolicy
from .prompting import SYSTEM_PROMPT, training_record
from .world import World, WorldConfig


def generate_sft_dataset(
    output_path: str | Path,
    world_config: WorldConfig,
    episodes: int,
    steps_per_episode: int,
    seed: int,
) -> dict[str, object]:
    if episodes < 1 or steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")

    records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for episode in range(episodes):
        episode_seed = seed + episode
        episode_records, episode_counts = _generate_episode_records(
            world_config,
            steps_per_episode,
            episode_seed,
        )
        records.extend(episode_records)
        action_counts.update(episode_counts)

    Random(seed).shuffle(records)
    destination = Path(output_path)
    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "records": len(records),
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "seed": seed,
        "action_counts": dict(sorted(action_counts.items())),
    }
    _write_dataset(destination, records, summary)
    return summary


def generate_balanced_sft_dataset(
    output_path: str | Path,
    base_dataset_path: str | Path,
    world_config: WorldConfig,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    minimum_per_action: int = 1_000,
) -> dict[str, object]:
    """Top up a natural SFT dataset with unique rare-action demonstrations."""

    if episodes < 1 or steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")
    if minimum_per_action < 1:
        raise ValueError("minimum_per_action must be positive")

    base_records = _read_sft_records(base_dataset_path)
    records: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    action_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    for record in base_records:
        fingerprint = _record_fingerprint(record)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        kind = _record_action_kind(record)
        records.append(record)
        action_counts[kind] += 1
        base_counts[kind] += 1

    candidate_counts: Counter[str] = Counter()
    added_counts: Counter[str] = Counter()
    episodes_used = 0
    action_names = tuple(kind.value for kind in ActionKind)
    for episode in range(episodes):
        episode_seed = seed + episode
        episode_records, episode_counts = _generate_episode_records(
            world_config,
            steps_per_episode,
            episode_seed,
        )
        episodes_used = episode + 1
        candidate_counts.update(episode_counts)
        for record in episode_records:
            kind = _record_action_kind(record)
            if action_counts[kind] >= minimum_per_action:
                continue
            fingerprint = _record_fingerprint(record)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            records.append(record)
            action_counts[kind] += 1
            added_counts[kind] += 1
        if all(action_counts[name] >= minimum_per_action for name in action_names):
            break

    missing = {
        name: minimum_per_action - action_counts[name]
        for name in action_names
        if action_counts[name] < minimum_per_action
    }
    if missing:
        raise ValueError(
            f"candidate generation did not reach minimum action counts after "
            f"{episodes_used} episodes: {missing}"
        )

    Random(seed).shuffle(records)
    destination = Path(output_path)
    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "base_dataset": str(Path(base_dataset_path).resolve()),
        "base_records": len(base_records),
        "base_unique_records": sum(base_counts.values()),
        "records": len(records),
        "candidate_seed": seed,
        "candidate_episodes_requested": episodes,
        "candidate_episodes_used": episodes_used,
        "steps_per_episode": steps_per_episode,
        "minimum_per_action": minimum_per_action,
        "base_action_counts": dict(sorted(base_counts.items())),
        "candidate_action_counts": dict(sorted(candidate_counts.items())),
        "added_action_counts": dict(sorted(added_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
    _write_dataset(destination, records, summary)
    return summary


def stratify_sft_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    records_per_action: int,
    seed: int,
) -> dict[str, object]:
    """Create a deterministic held-out set with equal coverage of every action."""

    if records_per_action < 1:
        raise ValueError("records_per_action must be positive")

    source_records = _read_sft_records(dataset_path)
    grouped: dict[str, list[dict[str, Any]]] = {
        kind.value: [] for kind in ActionKind
    }
    for record in source_records:
        grouped[_record_action_kind(record)].append(record)

    shortages = {
        kind: records_per_action - len(records)
        for kind, records in grouped.items()
        if len(records) < records_per_action
    }
    if shortages:
        raise ValueError(f"dataset lacks records for stratification: {shortages}")

    rng = Random(seed)
    selected: list[dict[str, Any]] = []
    for kind in sorted(grouped):
        candidates = list(grouped[kind])
        rng.shuffle(candidates)
        selected.extend(candidates[:records_per_action])
    rng.shuffle(selected)

    destination = Path(output_path)
    action_counts = Counter(_record_action_kind(record) for record in selected)
    source_counts = {kind: len(records) for kind, records in grouped.items()}
    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "source_dataset": str(Path(dataset_path).resolve()),
        "source_records": len(source_records),
        "records": len(selected),
        "records_per_action": records_per_action,
        "seed": seed,
        "source_action_counts": dict(sorted(source_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
    _write_dataset(destination, selected, summary)
    return summary


def generate_communication_sft_dataset(
    output_path: str | Path,
    world_config: WorldConfig,
    scenes: int,
    seed: int,
    base_dataset_path: str | Path | None = None,
) -> dict[str, object]:
    """Build deterministic sender/recipient counterfactuals, optionally atop an SFT corpus."""

    if scenes < 1:
        raise ValueError("scenes must be positive")
    if world_config.perception_radius < 1:
        raise ValueError("communication scenes require a positive perception radius")

    base_records = _read_sft_records(base_dataset_path) if base_dataset_path else []
    records: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    base_action_counts: Counter[str] = Counter()
    for source_record in base_records:
        record = _with_current_system_prompt(source_record, world_config.perception_radius)
        fingerprint = _training_fingerprint(record)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        records.append(record)
        base_action_counts[_record_action_kind(record)] += 1

    communication_case_counts: Counter[str] = Counter()
    communication_action_counts: Counter[str] = Counter()
    generated_scenes = 0
    rng = Random(seed)
    attempts = 0
    max_attempts = scenes * 100
    while generated_scenes < scenes and attempts < max_attempts:
        attempts += 1
        scene_records = _communication_scene_records(
            world_config=world_config,
            rng=rng,
            scene_number=generated_scenes,
            seed=seed,
        )
        if scene_records is None:
            continue
        if any(_training_fingerprint(record) in fingerprints for record in scene_records):
            continue
        for record in scene_records:
            fingerprints.add(_training_fingerprint(record))
            records.append(record)
            communication_case_counts[record["communication_case"]] += 1
            communication_action_counts[_record_action_kind(record)] += 1
        generated_scenes += 1

    if generated_scenes != scenes:
        raise ValueError(
            f"could generate only {generated_scenes} unique communication scenes "
            f"after {attempts} attempts"
        )

    Random(seed).shuffle(records)
    destination = Path(output_path)
    action_counts = Counter(_record_action_kind(record) for record in records)
    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "base_dataset": (
            str(Path(base_dataset_path).resolve()) if base_dataset_path is not None else None
        ),
        "base_records": len(base_records),
        "base_unique_records": sum(base_action_counts.values()),
        "communication_scenes": scenes,
        "communication_records": sum(communication_case_counts.values()),
        "records": len(records),
        "seed": seed,
        "perception_radius": world_config.perception_radius,
        "base_action_counts": dict(sorted(base_action_counts.items())),
        "communication_case_counts": dict(sorted(communication_case_counts.items())),
        "communication_action_counts": dict(sorted(communication_action_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
    _write_dataset(destination, records, summary)
    return summary


def _generate_episode_records(
    world_config: WorldConfig,
    steps_per_episode: int,
    episode_seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    world = World(world_config, seed=episode_seed)
    policy = DemonstrationPolicy(seed=episode_seed + 10_000)
    world.run(policy, steps_per_episode)
    records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for event in world.events:
        if event.kind != "decision" or not event.data.get("valid"):
            continue
        action = Action.from_payload(event.data["action"])
        records.append(training_record(event.data["observation"], action))
        action_counts[action.kind.value] += 1
    return records, action_counts


def _communication_scene_records(
    world_config: WorldConfig,
    rng: Random,
    scene_number: int,
    seed: int,
) -> list[dict[str, Any]] | None:
    radius = world_config.perception_radius
    sender = (
        rng.randrange(world_config.width),
        rng.randrange(world_config.height),
    )
    nearby = [
        position
        for position in _positions_within(sender, radius, world_config)
        if position != sender
    ]
    if not nearby:
        return None

    target_candidates = [
        position
        for position in nearby
        if _manhattan(sender, position) >= min(2, radius)
    ]
    if not target_candidates:
        return None
    target = rng.choice(target_candidates)
    useful_recipients = [
        position
        for position in nearby
        if position != target and _manhattan(position, target) > radius
    ]
    redundant_recipients = [
        position
        for position in nearby
        if position != target and _manhattan(position, target) <= radius
    ]
    if not useful_recipients or not redundant_recipients:
        return None

    useful_recipient = rng.choice(useful_recipients)
    redundant_recipient = rng.choice(redundant_recipients)
    sender_id = f"organism-s{seed}-{scene_number:05d}"
    recipient_id = f"organism-r{seed}-{scene_number:05d}"
    energy = rng.randint(7, 14)
    tick = rng.randint(2, 90)
    drives = {
        "reproduction_threshold": 16,
        "share_threshold": 12,
        "curiosity": round(rng.uniform(0.15, 0.65), 3),
        "sociality": round(rng.uniform(0.15, 0.75), 3),
        "fecundity": round(rng.uniform(0.1, 0.35), 3),
    }
    sender_memory = [f"last forage was {rng.randint(1, 8)} ticks ago"]
    target_amount = rng.randint(1, 3)
    recipient_energy = rng.randint(5, 11)

    sender_useful = _observation_payload(
        tick=tick,
        perception_radius=radius,
        organism_id=sender_id,
        lineage_id=f"lineage-s{scene_number % 17:02d}",
        position=sender,
        energy=energy,
        drives=drives,
        visible_food=[(target[0], target[1], target_amount)],
        visible_agents=[
            _visible_agent(recipient_id, useful_recipient, sender, energy=recipient_energy)
        ],
        open_neighbors=_open_neighbors(sender, world_config, {useful_recipient}),
        memory=sender_memory,
    )
    sender_redundant = _observation_payload(
        tick=tick,
        perception_radius=radius,
        organism_id=sender_id,
        lineage_id=f"lineage-s{scene_number % 17:02d}",
        position=sender,
        energy=energy,
        drives=drives,
        visible_food=[(target[0], target[1], target_amount)],
        visible_agents=[
            _visible_agent(recipient_id, redundant_recipient, sender, energy=recipient_energy)
        ],
        open_neighbors=_open_neighbors(sender, world_config, {redundant_recipient}),
        memory=sender_memory,
    )
    sender_move = _move_toward(
        target,
        [tuple(position) for position in sender_redundant["open_neighbors"]],
    )
    if sender_move is None:
        return None

    recipient_open = _open_neighbors(useful_recipient, world_config, {sender})
    recipient_move = _move_toward(target, recipient_open)
    if recipient_move is None:
        return None
    recipient_base = _observation_payload(
        tick=tick + 1,
        perception_radius=radius,
        organism_id=recipient_id,
        lineage_id=f"lineage-r{scene_number % 19:02d}",
        position=useful_recipient,
        energy=rng.randint(6, 14),
        drives=drives,
        visible_food=[],
        visible_agents=[_visible_agent(sender_id, sender, useful_recipient, energy=energy)],
        open_neighbors=recipient_open,
        memory=[f"survived tick {tick - 1}"],
    )
    recipient_informed = json.loads(json.dumps(recipient_base))
    recipient_informed["memory"].append(
        f"{sender_id} signalled: food at [{target[0]},{target[1]}]"
    )
    recipient_control = json.loads(json.dumps(recipient_base))

    sender_pair = f"sender-{seed}-{scene_number:05d}"
    recipient_pair = f"recipient-{seed}-{scene_number:05d}"
    return [
        _communication_record(
            sender_useful,
            Action(ActionKind.SIGNAL, message=f"food at [{target[0]},{target[1]}]"),
            case="sender_useful",
            pair_id=sender_pair,
            target_food=target,
        ),
        _communication_record(
            sender_redundant,
            Action(ActionKind.MOVE, target_position=sender_move),
            case="sender_redundant",
            pair_id=sender_pair,
            target_food=target,
        ),
        _communication_record(
            recipient_informed,
            Action(ActionKind.MOVE, target_position=recipient_move),
            case="recipient_informed",
            pair_id=recipient_pair,
            target_food=target,
        ),
        _communication_record(
            recipient_control,
            Action(ActionKind.REST),
            case="recipient_control",
            pair_id=recipient_pair,
            target_food=target,
        ),
    ]


def _communication_record(
    observation: dict[str, Any],
    action: Action,
    case: str,
    pair_id: str,
    target_food: Position,
) -> dict[str, Any]:
    record = training_record(observation, action)
    record.update(
        {
            "communication_case": case,
            "pair_id": pair_id,
            "target_food": list(target_food),
        }
    )
    return record


def _observation_payload(
    *,
    tick: int,
    perception_radius: int,
    organism_id: str,
    lineage_id: str,
    position: Position,
    energy: int,
    drives: dict[str, float | int],
    visible_food: list[tuple[int, int, int]],
    visible_agents: list[dict[str, Any]],
    open_neighbors: list[Position],
    memory: list[str],
) -> dict[str, Any]:
    return {
        "tick": tick,
        "perception_radius": perception_radius,
        "self": {
            "id": organism_id,
            "lineage": lineage_id,
            "position": list(position),
            "energy": energy,
            "age": tick,
            "drives": drives,
        },
        "current_food": 0,
        "visible_food": [
            {"position": [x, y], "amount": amount} for x, y, amount in visible_food
        ],
        "visible_agents": visible_agents,
        "open_neighbors": [list(position) for position in open_neighbors],
        "memory": memory,
    }


def _visible_agent(
    organism_id: str,
    position: Position,
    observer: Position,
    energy: int,
) -> dict[str, Any]:
    return {
        "id": organism_id,
        "lineage": f"lineage-{organism_id[-2:]}",
        "position": list(position),
        "energy": energy,
        "distance": _manhattan(observer, position),
    }


def _positions_within(
    origin: Position,
    radius: int,
    world_config: WorldConfig,
) -> list[Position]:
    return [
        (x, y)
        for x in range(world_config.width)
        for y in range(world_config.height)
        if _manhattan(origin, (x, y)) <= radius
    ]


def _open_neighbors(
    origin: Position,
    world_config: WorldConfig,
    occupied: set[Position],
) -> list[Position]:
    x, y = origin
    candidates = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
    return [
        position
        for position in candidates
        if 0 <= position[0] < world_config.width
        and 0 <= position[1] < world_config.height
        and position not in occupied
    ]


def _move_toward(
    target: Position,
    open_neighbors: list[Position],
) -> Position | None:
    if not open_neighbors:
        return None
    return min(
        open_neighbors,
        key=lambda position: (_manhattan(position, target), position),
    )


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _read_sft_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _record_action_kind(record)
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid SFT record on line {line_number}") from exc
            records.append(record)
    if not records:
        raise ValueError("SFT dataset is empty")
    return records


def _record_action_kind(record: dict[str, Any]) -> str:
    payload = json.loads(record["completion"][0]["content"])
    return Action.from_payload(payload).kind.value


def _record_fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _training_fingerprint(record: dict[str, Any]) -> str:
    return json.dumps(
        {"prompt": record["prompt"], "completion": record["completion"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _with_current_system_prompt(
    record: dict[str, Any],
    perception_radius: int,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(record))
    prompts = normalized.get("prompt", [])
    if not prompts or prompts[0].get("role") != "system":
        raise ValueError("SFT record has no leading system prompt")
    prompts[0]["content"] = SYSTEM_PROMPT
    for message in reversed(prompts):
        if message.get("role") != "user":
            continue
        observation = json.loads(message["content"])
        observation["perception_radius"] = perception_radius
        message["content"] = json.dumps(
            observation,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        break
    else:
        raise ValueError("SFT record has no user observation")
    return normalized


def _write_dataset(
    destination: Path,
    records: list[dict[str, Any]],
    summary: dict[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = destination.with_suffix(".summary.json")
    summary["summary_path"] = str(summary_path.resolve())
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

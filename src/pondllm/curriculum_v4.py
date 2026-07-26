from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

from .domain import Action, ActionKind, Genes, Observation, Position
from .prompting import training_record
from .world import World, WorldConfig


V4_COMMUNICATION_CASES = (
    "sender_useful",
    "sender_redundant",
    "recipient_informed",
    "recipient_control",
    "recipient_visible_override",
)
V4_SURVIVAL_CASES = (
    "share_safe",
    "share_unsafe_reserve",
    "share_not_needed",
    "low_energy_food_priority",
)

_FOOD_MEMORY = re.compile(r"food at \[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", re.IGNORECASE)


def generate_v4_sft_dataset(
    output_path: str | Path,
    world_config: WorldConfig,
    scenes: int,
    seed: int,
    base_dataset_path: str | Path | None = None,
    *,
    include_trajectories: bool = True,
    survival_scenes: int = 0,
) -> dict[str, object]:
    """Build a neutral, simulator-native communication and survival curriculum."""

    if scenes < 0:
        raise ValueError("scenes cannot be negative")
    if survival_scenes < 0:
        raise ValueError("survival_scenes cannot be negative")
    if scenes == 0 and survival_scenes == 0 and base_dataset_path is None:
        raise ValueError("V4 dataset requires scenes, survival scenes, or a base dataset")
    if world_config.perception_radius < 1:
        raise ValueError("V4 scenes require a positive perception radius")

    base_records = (
        _read_sft_records(base_dataset_path) if base_dataset_path is not None else []
    )
    records, base_summary = _curate_base_records(base_records)
    fingerprints = {_record_fingerprint(record) for record in records}
    communication_counts: Counter[str] = Counter()
    survival_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter(
        _record_action_kind(record) for record in records
    )
    rng = Random(seed)

    for scene_number in range(scenes):
        scene_records = _v4_communication_scene_records(
            world_config,
            rng,
            scene_number=scene_number,
            seed=seed,
            include_trajectory=include_trajectories,
        )
        for record in scene_records:
            fingerprint = _record_fingerprint(record)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            records.append(record)
            action_counts[_record_action_kind(record)] += 1
            communication_counts[str(record["communication_case"])] += 1

    for scene_number in range(survival_scenes):
        for record in _v4_survival_scene_records(
            world_config,
            rng,
            scene_number=scene_number,
            seed=seed,
        ):
            fingerprint = _record_fingerprint(record)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            records.append(record)
            action_counts[_record_action_kind(record)] += 1
            survival_counts[str(record["survival_case"])] += 1

    Random(seed).shuffle(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "curriculum": "v4-simulator-native",
        "seed": seed,
        "scenes": scenes,
        "include_trajectories": include_trajectories,
        "survival_scenes": survival_scenes,
        "base_dataset": (
            str(Path(base_dataset_path).resolve())
            if base_dataset_path is not None
            else None
        ),
        "base": base_summary,
        "communication_case_counts": dict(sorted(communication_counts.items())),
        "survival_case_counts": dict(sorted(survival_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "records": len(records),
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _curate_base_records(
    base_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    records: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    kept: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    for source_record in base_records:
        observation = _observation_from_record(source_record)
        action = Action.from_payload(
            json.loads(source_record["completion"][0]["content"])
        )
        if action.kind is ActionKind.SIGNAL:
            removed["generic_signal"] += 1
            continue
        if action.kind is ActionKind.SHARE and not _safe_share_example(
            observation, action
        ):
            removed["unsafe_or_unneeded_share"] += 1
            continue
        record = training_record(observation, action)
        fingerprint = _record_fingerprint(record)
        if fingerprint in fingerprints:
            removed["duplicate"] += 1
            continue
        fingerprints.add(fingerprint)
        records.append(record)
        kept[action.kind.value] += 1
    return records, {
        "source_records": len(base_records),
        "kept_records": len(records),
        "kept_action_counts": dict(sorted(kept.items())),
        "removed_counts": dict(sorted(removed.items())),
    }


def _safe_share_example(observation: dict[str, Any], action: Action) -> bool:
    self_state = observation["self"]
    threshold = int(
        self_state.get("drives", {}).get("share_threshold", 10**9)
    )
    if int(self_state.get("energy", 0)) < threshold:
        return False
    candidates = {
        str(agent["id"]): agent
        for agent in observation.get("visible_agents", [])
        if int(agent.get("distance", -1)) == 1
    }
    target = candidates.get(str(action.target_id))
    return target is not None and int(target.get("energy", 10**9)) <= 3


def _v4_communication_scene_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    scene_number: int,
    seed: int,
    include_trajectory: bool,
) -> list[dict[str, Any]]:
    sender_position, useful_recipient, redundant_recipient, target = _sample_layout(
        world_config, rng
    )
    identity = _sample_identity(rng, seed, scene_number)
    state = _sample_state(rng, seed, scene_number, identity, target)

    useful_world, sender_id, recipient_id = _build_scene_world(
        world_config,
        sender_position,
        useful_recipient,
        target,
        identity,
        state,
    )
    redundant_world, redundant_sender_id, redundant_recipient_id = _build_scene_world(
        world_config,
        sender_position,
        redundant_recipient,
        target,
        identity,
        state,
    )
    if sender_id != redundant_sender_id or recipient_id != redundant_recipient_id:
        raise RuntimeError("paired V4 worlds changed neutral identities")

    scene_id = f"v4-scene-{seed}-{scene_number:05d}"
    signal = Action(
        ActionKind.SIGNAL,
        message=f"food at [{target[0]},{target[1]}]",
    )
    sender_useful = useful_world.observe(useful_world.organisms[sender_id])
    sender_redundant = redundant_world.observe(
        redundant_world.organisms[sender_id]
    )
    sender_move = _move_toward(target, sender_redundant.open_neighbors)
    if sender_move is None:
        raise RuntimeError("V4 sender has no move toward visible food")

    recipient_control = useful_world.observe(
        useful_world.organisms[recipient_id]
    )
    control_action = _local_control_action(recipient_control)
    valid, _ = useful_world.apply_action(useful_world.organisms[sender_id], signal)
    if not valid:
        raise RuntimeError("V4 sender signal was unexpectedly invalid")
    recipient_informed = useful_world.observe(
        useful_world.organisms[recipient_id]
    )
    informed_move = _move_toward(target, recipient_informed.open_neighbors)
    if informed_move is None:
        raise RuntimeError("V4 informed recipient has no move toward target")

    records = [
        _communication_record(
            sender_useful,
            signal,
            case="sender_useful",
            pair_id=f"{scene_id}-sender",
            scene_id=scene_id,
            target_food=target,
        ),
        _communication_record(
            sender_redundant,
            Action(ActionKind.MOVE, target_position=sender_move),
            case="sender_redundant",
            pair_id=f"{scene_id}-sender",
            scene_id=scene_id,
            target_food=target,
        ),
        _communication_record(
            recipient_informed,
            Action(ActionKind.MOVE, target_position=informed_move),
            case="recipient_informed",
            pair_id=f"{scene_id}-recipient",
            scene_id=scene_id,
            target_food=target,
        ),
        _communication_record(
            recipient_control,
            control_action,
            case="recipient_control",
            pair_id=f"{scene_id}-recipient",
            scene_id=scene_id,
            target_food=target,
        ),
    ]

    wrong_target = _wrong_coordinate(target, world_config.width, world_config.height)
    wrong_signal = Action(
        ActionKind.SIGNAL,
        message=f"food at [{wrong_target[0]},{wrong_target[1]}]",
    )
    valid, _ = redundant_world.apply_action(
        redundant_world.organisms[sender_id], wrong_signal
    )
    if not valid:
        raise RuntimeError("V4 visible-override signal was unexpectedly invalid")
    visible_override = redundant_world.observe(
        redundant_world.organisms[recipient_id]
    )
    override_move = _move_toward(target, visible_override.open_neighbors)
    if override_move is None:
        raise RuntimeError("V4 visible-override recipient cannot approach food")
    records.append(
        _communication_record(
            visible_override,
            Action(ActionKind.MOVE, target_position=override_move),
            case="recipient_visible_override",
            pair_id=f"{scene_id}-override",
            scene_id=scene_id,
            target_food=target,
        )
    )

    if include_trajectory:
        records.extend(
            _trajectory_records(
                world_config,
                sender_position,
                useful_recipient,
                target,
                identity,
                state,
                scene_id,
            )
        )
    return records


def _trajectory_records(
    world_config: WorldConfig,
    sender_position: Position,
    recipient_position: Position,
    target: Position,
    identity: dict[str, str],
    state: dict[str, Any],
    scene_id: str,
) -> list[dict[str, Any]]:
    trajectory_state = dict(state)
    trajectory_state["recipient_energy"] = max(
        int(state["recipient_energy"]),
        world_config.perception_radius * 2 + 7,
    )
    world, sender_id, recipient_id = _build_scene_world(
        world_config,
        sender_position,
        recipient_position,
        target,
        identity,
        trajectory_state,
    )
    policy = _V4TrajectoryPolicy(sender_id, recipient_id, target)
    world.run(policy, steps=world.config.perception_radius * 2 + 5)
    records: list[dict[str, Any]] = []
    step = 0
    for event in world.events:
        if event.kind != "decision" or event.actor_id != recipient_id:
            continue
        observation = event.data["observation"]
        if not any(
            f"food at [{target[0]},{target[1]}]" in item
            for item in observation["memory"]
        ):
            continue
        action = Action.from_payload(event.data["action"])
        record = _communication_record(
            observation,
            action,
            case="recipient_trajectory",
            pair_id=f"{scene_id}-trajectory-{step:02d}",
            scene_id=scene_id,
            target_food=target,
        )
        record["trajectory_step"] = step
        records.append(record)
        step += 1
        if action.kind is ActionKind.FORAGE:
            break
    if not records or _record_action_kind(records[-1]) != ActionKind.FORAGE.value:
        raise RuntimeError("V4 trajectory did not reach and forage target food")
    return records


def _v4_survival_scene_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    scene_number: int,
    seed: int,
) -> list[dict[str, Any]]:
    x = rng.randrange(1, world_config.width - 1)
    y = rng.randrange(1, world_config.height - 1)
    donor_position = (x, y)
    recipient_position = rng.choice(
        ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
    )
    food_candidates = [
        position
        for position in _positions_within(
            donor_position, world_config.perception_radius, world_config
        )
        if position not in {donor_position, recipient_position}
        and _manhattan(position, donor_position) >= 2
    ]
    target_food = rng.choice(food_candidates)
    identity = _sample_identity(rng, seed + 1_000_000, scene_number)
    genes = _sample_genes(rng)
    tick = rng.randint(0, 120)
    state = {
        "world_seed": (seed + 1_000_000) * 100_000 + scene_number,
        "tick": tick,
        "spawn_sender_first": bool(rng.randrange(2)),
        "sender_age": rng.randint(0, tick),
        "recipient_age": rng.randint(0, tick),
        "sender_energy": genes.share_threshold + rng.randint(3, 7),
        "recipient_energy": 2,
        "sender_genes": genes,
        "recipient_genes": _sample_genes(rng),
        "sender_memory": _natural_memory(rng, identity["recipient_id"], target_food),
        "recipient_memory": _natural_memory(rng, identity["sender_id"], target_food),
    }
    scene_id = f"v4-survival-{seed}-{scene_number:05d}"
    records: list[dict[str, Any]] = []

    safe_world, donor_id, recipient_id = _build_scene_world(
        world_config,
        donor_position,
        recipient_position,
        target_food,
        identity,
        state,
        target_amount=0,
    )
    safe_observation = safe_world.observe(safe_world.organisms[donor_id])
    records.append(
        _survival_record(
            safe_observation,
            Action(ActionKind.SHARE, target_id=recipient_id, amount=1),
            "share_safe",
            scene_id,
        )
    )

    unsafe_state = dict(state)
    unsafe_state["sender_energy"] = max(3, genes.share_threshold - rng.randint(1, 3))
    unsafe_world, unsafe_donor_id, _ = _build_scene_world(
        world_config,
        donor_position,
        recipient_position,
        target_food,
        identity,
        unsafe_state,
        target_amount=0,
    )
    records.append(
        _survival_record(
            unsafe_world.observe(unsafe_world.organisms[unsafe_donor_id]),
            Action(ActionKind.REST),
            "share_unsafe_reserve",
            scene_id,
        )
    )

    not_needed_state = dict(state)
    not_needed_state["recipient_energy"] = rng.randint(7, 12)
    not_needed_world, not_needed_donor_id, _ = _build_scene_world(
        world_config,
        donor_position,
        recipient_position,
        target_food,
        identity,
        not_needed_state,
        target_amount=0,
    )
    records.append(
        _survival_record(
            not_needed_world.observe(
                not_needed_world.organisms[not_needed_donor_id]
            ),
            Action(ActionKind.REST),
            "share_not_needed",
            scene_id,
        )
    )

    food_state = dict(state)
    food_state["sender_energy"] = max(3, genes.share_threshold - 2)
    food_world, food_donor_id, _ = _build_scene_world(
        world_config,
        donor_position,
        recipient_position,
        target_food,
        identity,
        food_state,
        target_amount=2,
    )
    food_observation = food_world.observe(food_world.organisms[food_donor_id])
    food_move = _move_toward(target_food, food_observation.open_neighbors)
    if food_move is None:
        raise RuntimeError("V4 low-energy donor cannot approach visible food")
    records.append(
        _survival_record(
            food_observation,
            Action(ActionKind.MOVE, target_position=food_move),
            "low_energy_food_priority",
            scene_id,
        )
    )
    return records


def _build_scene_world(
    world_config: WorldConfig,
    sender_position: Position,
    recipient_position: Position,
    target_food: Position,
    identity: dict[str, str],
    state: dict[str, Any],
    *,
    target_amount: int = 2,
) -> tuple[World, str, str]:
    config = replace(
        world_config,
        founders=2,
        initial_food=0,
        food_regrowth=0,
        max_food=max(4, world_config.max_food),
        max_population=max(4, world_config.founders),
        signal_delivery="normal",
        signal_cost=0,
    )
    world = World(config, seed=int(state["world_seed"]), initialize=False)
    world.tick = int(state["tick"])
    spawn_sender_first = bool(state["spawn_sender_first"])
    roles = (
        ("sender", sender_position),
        ("recipient", recipient_position),
    )
    if not spawn_sender_first:
        roles = tuple(reversed(roles))
    organisms: dict[str, Any] = {}
    for role, position in roles:
        organism = world.spawn_founder(
            position=position,
            lineage_id=identity[f"{role}_lineage"],
            energy=int(state[f"{role}_energy"]),
            genes=state[f"{role}_genes"],
            organism_id=identity[f"{role}_id"],
        )
        organism.age = int(state[f"{role}_age"])
        for item in state[f"{role}_memory"]:
            organism.remember(item, world.config.memory_limit)
        organisms[role] = organism
    if target_amount > 0:
        world.add_food(target_food, target_amount)
    return world, organisms["sender"].organism_id, organisms["recipient"].organism_id


def _sample_identity(
    rng: Random,
    seed: int,
    scene_number: int,
) -> dict[str, str]:
    prefix = abs(seed) % 1_000_000
    organism_ids = [
        f"organism-{prefix:06d}-{scene_number * 2 + offset:06d}"
        for offset in (1, 2)
    ]
    lineages = [
        f"lineage-{(prefix * 31 + scene_number * 2 + offset) % 10_000_000:07d}"
        for offset in (1, 2)
    ]
    rng.shuffle(organism_ids)
    rng.shuffle(lineages)
    return {
        "sender_id": organism_ids[0],
        "recipient_id": organism_ids[1],
        "sender_lineage": lineages[0],
        "recipient_lineage": lineages[1],
    }


def _sample_state(
    rng: Random,
    seed: int,
    scene_number: int,
    identity: dict[str, str],
    target: Position,
) -> dict[str, Any]:
    tick = rng.randint(0, 120)
    sender_genes = _sample_genes(rng)
    recipient_genes = _sample_genes(rng)
    return {
        "world_seed": seed * 100_000 + scene_number,
        "tick": tick,
        "spawn_sender_first": bool(rng.randrange(2)),
        "sender_age": rng.randint(0, tick),
        "recipient_age": rng.randint(0, tick),
        "sender_energy": rng.randint(7, 15),
        "recipient_energy": rng.randint(8, 14),
        "sender_genes": sender_genes,
        "recipient_genes": recipient_genes,
        "sender_memory": _natural_memory(rng, identity["recipient_id"], target),
        "recipient_memory": _natural_memory(rng, identity["sender_id"], target),
    }


def _sample_genes(rng: Random) -> Genes:
    return Genes(
        reproduction_threshold=rng.randint(14, 20),
        share_threshold=rng.randint(10, 15),
        curiosity=round(rng.uniform(0.15, 0.7), 3),
        sociality=round(rng.uniform(0.1, 0.75), 3),
        fecundity=round(rng.uniform(0.1, 0.35), 3),
    )


def _natural_memory(
    rng: Random,
    other_id: str,
    target: Position,
) -> list[str]:
    mode = rng.randrange(4)
    if mode == 0:
        return []
    if mode == 1:
        return [f"{other_id} shared 1 energy"]
    wrong = (
        target[0] - 1 if target[0] > 0 else target[0] + 1,
        target[1] - 1 if target[1] > 0 else target[1] + 1,
    )
    signal = f"{other_id} signalled: food at [{wrong[0]},{wrong[1]}]"
    if mode == 2:
        return [signal]
    return [f"{other_id} shared 1 energy", signal]


def _sample_layout(
    world_config: WorldConfig,
    rng: Random,
) -> tuple[Position, Position, Position, Position]:
    radius = world_config.perception_radius
    for _ in range(10_000):
        sender = (
            rng.randrange(world_config.width),
            rng.randrange(world_config.height),
        )
        nearby = [
            position
            for position in _positions_within(sender, radius, world_config)
            if position != sender
        ]
        targets = [
            position
            for position in nearby
            if _manhattan(sender, position) >= min(2, radius)
        ]
        if not targets:
            continue
        target = rng.choice(targets)
        useful = [
            position
            for position in nearby
            if position != target and _manhattan(position, target) > radius
        ]
        redundant = [
            position
            for position in nearby
            if position != target and _manhattan(position, target) <= radius
        ]
        if useful and redundant:
            return sender, rng.choice(useful), rng.choice(redundant), target
    raise RuntimeError("could not construct a V4 communication layout")


def _positions_within(
    origin: Position,
    radius: int,
    world_config: WorldConfig,
) -> list[Position]:
    positions = []
    for x in range(max(0, origin[0] - radius), min(world_config.width, origin[0] + radius + 1)):
        for y in range(
            max(0, origin[1] - radius),
            min(world_config.height, origin[1] + radius + 1),
        ):
            position = (x, y)
            if _manhattan(origin, position) <= radius:
                positions.append(position)
    return positions


def _local_control_action(observation: Observation) -> Action:
    if observation.current_food > 0:
        return Action(ActionKind.FORAGE)
    if observation.visible_food:
        target = observation.visible_food[0][:2]
        move = _move_toward(target, observation.open_neighbors)
        if move is not None:
            return Action(ActionKind.MOVE, target_position=move)
    remembered = _remembered_food(observation.memory)
    if remembered is not None:
        move = _move_toward(remembered, observation.open_neighbors)
        if move is not None:
            return Action(ActionKind.MOVE, target_position=move)
    if (
        observation.energy > 7
        and float(observation.drives["curiosity"]) >= 0.5
        and observation.open_neighbors
    ):
        return Action(
            ActionKind.MOVE,
            target_position=min(observation.open_neighbors),
        )
    return Action(ActionKind.REST)


def _remembered_food(memory: tuple[str, ...] | list[str]) -> Position | None:
    for item in reversed(memory):
        match = _FOOD_MEMORY.search(item)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _move_toward(
    target: Position,
    open_neighbors: tuple[Position, ...] | list[Position],
) -> Position | None:
    if not open_neighbors:
        return None
    best_distance = min(_manhattan(position, target) for position in open_neighbors)
    return min(
        position
        for position in open_neighbors
        if _manhattan(position, target) == best_distance
    )


def _wrong_coordinate(target: Position, width: int, height: int) -> Position:
    return (
        (target[0] + max(2, width // 2)) % width,
        (target[1] + max(2, height // 2)) % height,
    )


def _communication_record(
    observation: Observation | dict[str, Any],
    action: Action,
    *,
    case: str,
    pair_id: str,
    scene_id: str,
    target_food: Position,
) -> dict[str, Any]:
    record = training_record(observation, action)
    record.update(
        {
            "curriculum": "v4",
            "communication_case": case,
            "pair_id": pair_id,
            "scene_id": scene_id,
            "target_food": list(target_food),
        }
    )
    return record


def _survival_record(
    observation: Observation,
    action: Action,
    case: str,
    scene_id: str,
) -> dict[str, Any]:
    record = training_record(observation, action)
    record.update(
        {
            "curriculum": "v4",
            "survival_case": case,
            "scene_id": scene_id,
        }
    )
    return record


def _observation_from_record(record: dict[str, Any]) -> dict[str, Any]:
    for message in record.get("prompt", []):
        if message.get("role") == "user":
            payload = json.loads(message["content"])
            if isinstance(payload, dict):
                return payload
    raise ValueError("SFT record has no object-valued user observation")


def _record_action_kind(record: dict[str, Any]) -> str:
    return str(json.loads(record["completion"][0]["content"])["action"])


def _record_fingerprint(record: dict[str, Any]) -> str:
    payload = {
        "prompt": record["prompt"],
        "completion": record["completion"],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_sft_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid SFT record on line {line_number}")
            records.append(record)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


class _V4TrajectoryPolicy:
    def __init__(
        self,
        sender_id: str,
        recipient_id: str,
        target: Position,
    ) -> None:
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.target = target
        self.sent = False
        self.recipient_visited: set[Position] = set()

    def choose(self, observation: Observation) -> Action:
        if observation.organism_id == self.sender_id:
            if not self.sent:
                self.sent = True
                return Action(
                    ActionKind.SIGNAL,
                    message=f"food at [{self.target[0]},{self.target[1]}]",
                )
            return Action(ActionKind.REST)
        if observation.organism_id != self.recipient_id:
            return Action(ActionKind.REST)
        self.recipient_visited.add(observation.position)
        if observation.current_food > 0:
            return Action(ActionKind.FORAGE)
        remembered = _remembered_food(observation.memory)
        if remembered == self.target:
            move = _trajectory_move(
                observation,
                self.target,
                self.recipient_visited,
            )
            if move is not None:
                return Action(ActionKind.MOVE, target_position=move)
        return Action(ActionKind.REST)


def _trajectory_move(
    observation: Observation,
    target: Position,
    visited: set[Position],
) -> Position | None:
    if not observation.open_neighbors:
        return None
    occupied = {
        tuple(agent["position"]) for agent in observation.visible_agents
    }

    def score(candidate: Position) -> tuple[int, int, int, Position]:
        future = [
            position
            for position in (
                (candidate[0] - 1, candidate[1]),
                (candidate[0] + 1, candidate[1]),
                (candidate[0], candidate[1] - 1),
                (candidate[0], candidate[1] + 1),
            )
            if position[0] >= 0 and position[1] >= 0 and position not in occupied
        ]
        future_distance = min(
            (_manhattan(position, target) for position in future),
            default=10**9,
        )
        return (
            _manhattan(candidate, target),
            future_distance,
            int(candidate in visited),
            candidate,
        )

    return min(observation.open_neighbors, key=score)

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

from .curriculum_v4 import (
    _local_control_action,
    _manhattan,
    _move_toward,
    _observation_from_record,
    _read_sft_records,
    _record_action_kind,
    _record_fingerprint,
    _sample_genes,
    _sha256,
)
from .curriculum_v41 import (
    _audit_passes,
    _audit_records,
    _normalize_record,
    _rich_communication_records,
    _sample_reachability_identity,
    _sample_reachability_layout,
    _semantic_fingerprint,
)
from .domain import Action, ActionKind, Genes, Observation, Position
from .prompting import training_record
from .world import World, WorldConfig


V42_ENERGY_CASES = (
    "boundary_one_wait",
    "boundary_two_wait",
    "one_above_one_move",
    "one_above_one_forage",
    "one_above_two_move_1",
    "one_above_two_move_2",
    "one_above_two_forage",
    "energy_one_current_forage",
    "rescue_share_safe",
    "rescue_child_move",
    "rescue_child_forage",
    "rescue_share_not_needed",
    "rescue_share_not_needed_at_food",
    "rescue_share_unsafe",
    "unrescued_child_wait",
)


def generate_v42_sft_dataset(
    output_path: str | Path,
    world_config: WorldConfig,
    paired_scenes: int,
    redundant_scenes: int,
    trajectory_scenes: int,
    seed: int,
    base_dataset_path: str | Path | None = None,
    mined_predictions_path: str | Path | None = None,
) -> dict[str, object]:
    """Build the V4.2 hard-negative and sequence-level SFT correction."""

    counts = (paired_scenes, redundant_scenes, trajectory_scenes)
    if any(count < 0 for count in counts):
        raise ValueError("V4.2 scene counts cannot be negative")
    if not any(counts) and base_dataset_path is None and mined_predictions_path is None:
        raise ValueError("V4.2 requires generated scenes, a base dataset, or mined errors")
    if world_config.perception_radius < 2:
        raise ValueError("V4.2 scenes require perception radius of at least two")
    if (
        world_config.move_cost != 0
        or world_config.metabolism != 1
        or world_config.rest_gain != 1
    ):
        raise ValueError(
            "V4.2 energy labels are calibrated to move_cost=0, metabolism=1, rest_gain=1"
        )

    source_records = (
        _read_sft_records(base_dataset_path)
        if base_dataset_path is not None
        else []
    )
    records = [_normalize_record(record) for record in source_records]
    fingerprints = {_record_fingerprint(record) for record in records}
    if len(fingerprints) != len(records):
        raise ValueError("V4.2 base dataset contains duplicate prompt/completion records")

    action_counts: Counter[str] = Counter(
        _record_action_kind(record) for record in records
    )
    communication_counts: Counter[str] = Counter()
    energy_counts: Counter[str] = Counter()
    duplicate_records_skipped = 0
    mined_errors = 0
    rng = Random(seed)

    def add(record: dict[str, Any]) -> None:
        nonlocal duplicate_records_skipped
        fingerprint = _record_fingerprint(record)
        if fingerprint in fingerprints:
            duplicate_records_skipped += 1
            return
        fingerprints.add(fingerprint)
        records.append(record)
        action_counts[_record_action_kind(record)] += 1
        if "communication_case" in record:
            communication_counts[str(record["communication_case"])] += 1
        if "energy_case" in record:
            energy_counts[str(record["energy_case"])] += 1

    if mined_predictions_path is not None:
        for record in _mined_redundant_errors(mined_predictions_path):
            before = len(records)
            add(record)
            mined_errors += int(len(records) > before)

    for scene_number in range(paired_scenes):
        scene_records = _rich_communication_records(
            world_config,
            rng,
            seed=seed,
            scene_number=scene_number,
            include_trajectory=False,
        )
        for record in scene_records:
            if record.get("communication_case") not in {
                "rich_sender_useful",
                "rich_sender_redundant",
            }:
                continue
            add(_mark_hard_communication(record, source="paired"))

    redundant_offset = paired_scenes
    for scene_number in range(redundant_scenes):
        scene_records = _rich_communication_records(
            world_config,
            rng,
            seed=seed,
            scene_number=redundant_offset + scene_number,
            include_trajectory=False,
        )
        redundant = next(
            record
            for record in scene_records
            if record.get("communication_case") == "rich_sender_redundant"
        )
        add(_mark_hard_communication(redundant, source="redundant-oversample"))

    trajectory_offset = paired_scenes + redundant_scenes
    for scene_number in range(trajectory_scenes):
        for record in _energy_trajectory_records(
            world_config,
            rng,
            seed=seed,
            scene_number=trajectory_offset + scene_number,
        ):
            add(record)

    Random(seed).shuffle(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "curriculum": "v4.2-hard-negative-trajectories",
        "seed": seed,
        "base_dataset": (
            str(Path(base_dataset_path).resolve())
            if base_dataset_path is not None
            else None
        ),
        "base_records": len(source_records),
        "base_prompt_records_normalized": len(source_records),
        "mined_predictions": (
            str(Path(mined_predictions_path).resolve())
            if mined_predictions_path is not None
            else None
        ),
        "mined_redundant_errors": mined_errors,
        "paired_scenes": paired_scenes,
        "redundant_scenes": redundant_scenes,
        "trajectory_scenes": trajectory_scenes,
        "communication_case_counts": dict(sorted(communication_counts.items())),
        "energy_case_counts": dict(sorted(energy_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_records_skipped": duplicate_records_skipped,
        "unique_prompt_completion_records": len(fingerprints),
        "records": len(records),
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def audit_v42_datasets(
    training_path: str | Path,
    held_out_paths: list[str | Path],
) -> dict[str, object]:
    """Audit legality, energy annotations, and disjoint held-out examples."""

    training_records = _read_sft_records(training_path)
    held_out = {
        str(Path(path).resolve()): _read_sft_records(path)
        for path in held_out_paths
    }
    training_fingerprints = {
        _semantic_fingerprint(record) for record in training_records
    }
    training_audit = _audit_v42_records(training_records)
    held_out_audits = {
        path: _audit_v42_records(records) for path, records in held_out.items()
    }
    overlaps = {
        path: len(
            training_fingerprints
            & {_semantic_fingerprint(record) for record in records}
        )
        for path, records in held_out.items()
    }
    return {
        "training": str(Path(training_path).resolve()),
        "training_sha256": _sha256(Path(training_path)),
        "training_audit": training_audit,
        "held_out_audits": held_out_audits,
        "training_held_out_prompt_completion_overlap": overlaps,
        "all_checks_pass": (
            _audit_passes(training_audit)
            and training_audit["v42_energy_annotation_errors"] == 0
            and all(
                _audit_passes(audit)
                and audit["v42_energy_annotation_errors"] == 0
                for audit in held_out_audits.values()
            )
            and not any(overlaps.values())
        ),
    }


def _audit_v42_records(records: list[dict[str, Any]]) -> dict[str, int]:
    audit = _audit_records(records)
    annotation_errors = 0
    for record in records:
        if record.get("energy_case") not in V42_ENERGY_CASES:
            continue
        observation = _observation_from_record(record)
        target = tuple(record["target_food"])
        distance = _manhattan(tuple(observation["self"]["position"]), target)
        expected = bool(observation["current_food"]) or (
            int(observation["self"]["energy"]) > distance
        )
        annotation_errors += int(
            int(record["energy_budget_required"]) != distance + 1
            or bool(record["reachable_before_exhaustion"]) != expected
        )
    audit["v42_energy_annotation_errors"] = annotation_errors
    return audit


def _mined_redundant_errors(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            prediction = json.loads(line)
            if (
                prediction.get("communication_case") != "rich_sender_redundant"
                or prediction.get("predicted", {}).get("action") != "signal"
            ):
                continue
            action = Action.from_payload(prediction["expected"])
            record = training_record(prediction["observation"], action)
            target = prediction["target_food"]
            record.update(
                {
                    "curriculum": "v4.2",
                    "communication_case": "rich_sender_redundant",
                    "pair_id": prediction["pair_id"],
                    "scene_id": f"mined-{prediction['pair_id']}",
                    "target_food": target,
                    "context_agents": len(
                        prediction["observation"].get("visible_agents", [])
                    )
                    + 1,
                    "visible_food_count": len(
                        prediction["observation"].get("visible_food", [])
                    ),
                    "hard_negative_source": "v4.1-redundant-signal-error",
                }
            )
            records.append(_mark_hard_communication(record, source="mined-error"))
    return records


def _mark_hard_communication(
    source_record: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    record = dict(source_record)
    observation = _observation_from_record(record)
    target = tuple(record["target_food"])
    radius = int(observation["perception_radius"])
    target_informed = sum(
        _manhattan(tuple(agent["position"]), target) <= radius
        for agent in observation["visible_agents"]
    )
    visible_food = [
        tuple(food["position"]) for food in observation.get("visible_food", [])
    ]
    resource_informed = sum(
        any(
            _manhattan(tuple(agent["position"]), food) <= radius
            for food in visible_food
        )
        for agent in observation["visible_agents"]
    )
    resource_uninformed = len(observation["visible_agents"]) - resource_informed
    case = record.get("communication_case")
    if case == "rich_sender_useful" and resource_uninformed < 1:
        raise RuntimeError("V4.2 useful signal has no resource-uninformed peer")
    if case == "rich_sender_redundant" and resource_uninformed:
        raise RuntimeError("V4.2 redundant signal case still has a resource-uninformed peer")
    record.update(
        {
            "curriculum": "v4.2",
            "hard_negative_source": source,
            "peers_with_target_in_view": target_informed,
            "peers_without_target_in_view": (
                len(observation["visible_agents"]) - target_informed
            ),
            "peers_with_any_food_in_view": resource_informed,
            "peers_without_any_food_in_view": resource_uninformed,
        }
    )
    return record


def _energy_trajectory_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    seed: int,
    scene_number: int,
) -> list[dict[str, Any]]:
    actor, _, bystanders = _sample_reachability_layout(world_config, rng)
    distant_target = _distance_two_target(actor, bystanders, world_config, rng)
    identity = _sample_reachability_identity(rng, seed, scene_number)
    genes = _sample_genes(rng)
    tick = rng.randint(4, 160)
    scene_id = f"v42-energy-{seed}-{scene_number:05d}"
    one_target = _adjacent_target(actor, bystanders, world_config)
    records: list[dict[str, Any]] = []

    boundary_one, boundary_one_id = _build_energy_world(
        world_config,
        actor,
        one_target,
        bystanders,
        identity,
        genes,
        energy=1,
        seed=seed * 400_000 + scene_number * 20,
        tick=tick,
        memory=_failure_memory(rng, identity, one_target),
    )
    records.append(
        _energy_record(
            boundary_one.observe(boundary_one.organisms[boundary_one_id]),
            Action(ActionKind.REST),
            "boundary_one_wait",
            scene_id,
            one_target,
            False,
            trajectory_step=0,
        )
    )

    distance = _manhattan(actor, distant_target)
    boundary_two, boundary_two_id = _build_energy_world(
        world_config,
        actor,
        distant_target,
        bystanders,
        identity,
        genes,
        energy=distance,
        seed=seed * 400_000 + scene_number * 20 + 1,
        tick=tick,
        memory=_failure_memory(rng, identity, distant_target),
    )
    records.append(
        _energy_record(
            boundary_two.observe(boundary_two.organisms[boundary_two_id]),
            Action(ActionKind.REST),
            "boundary_two_wait",
            scene_id,
            distant_target,
            False,
            trajectory_step=0,
        )
    )

    one_above, one_above_id = _build_energy_world(
        world_config,
        actor,
        one_target,
        bystanders,
        identity,
        genes,
        energy=2,
        seed=seed * 400_000 + scene_number * 20 + 2,
        tick=tick,
        memory=_failure_memory(rng, identity, one_target),
    )
    one_actor = one_above.organisms[one_above_id]
    first_observation = one_above.observe(one_actor)
    move = _required_move(first_observation, one_target)
    records.append(
        _energy_record(
            first_observation,
            move,
            "one_above_one_move",
            scene_id,
            one_target,
            True,
            trajectory_step=0,
        )
    )
    _apply_turn(one_above, one_actor, move)
    records.append(
        _energy_record(
            one_above.observe(one_actor),
            Action(ActionKind.FORAGE),
            "one_above_one_forage",
            scene_id,
            one_target,
            True,
            trajectory_step=1,
        )
    )

    two_above, two_above_id = _build_energy_world(
        world_config,
        actor,
        distant_target,
        bystanders,
        identity,
        genes,
        energy=distance + 1,
        seed=seed * 400_000 + scene_number * 20 + 3,
        tick=tick,
        memory=_failure_memory(rng, identity, distant_target),
    )
    two_actor = two_above.organisms[two_above_id]
    step = 0
    while two_actor.position != distant_target:
        observation = two_above.observe(two_actor)
        move = _required_move(observation, distant_target)
        records.append(
            _energy_record(
                observation,
                move,
                f"one_above_two_move_{step + 1}",
                scene_id,
                distant_target,
                True,
                trajectory_step=step,
            )
        )
        _apply_turn(two_above, two_actor, move)
        step += 1
    records.append(
        _energy_record(
            two_above.observe(two_actor),
            Action(ActionKind.FORAGE),
            "one_above_two_forage",
            scene_id,
            distant_target,
            True,
            trajectory_step=step,
        )
    )

    current, current_id = _build_energy_world(
        world_config,
        actor,
        actor,
        bystanders,
        identity,
        genes,
        energy=1,
        seed=seed * 400_000 + scene_number * 20 + 4,
        tick=tick,
        memory=_failure_memory(rng, identity, one_target),
    )
    records.append(
        _energy_record(
            current.observe(current.organisms[current_id]),
            Action(ActionKind.FORAGE),
            "energy_one_current_forage",
            scene_id,
            actor,
            True,
            trajectory_step=0,
        )
    )
    records.extend(
        _rescue_trajectory_records(
            world_config,
            rng,
            actor=actor,
            target=one_target,
            bystanders=bystanders,
            identity=identity,
            donor_genes=genes,
            seed=seed * 400_000 + scene_number * 20 + 5,
            tick=tick,
            scene_id=scene_id,
        )
    )
    return records


def _rescue_trajectory_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    actor: Position,
    target: Position,
    bystanders: tuple[Position, Position],
    identity: dict[str, str],
    donor_genes: Genes,
    seed: int,
    tick: int,
    scene_id: str,
) -> list[dict[str, Any]]:
    safe, ids = _build_rescue_world(
        world_config,
        actor,
        target,
        bystanders,
        identity,
        donor_genes,
        child_energy=1,
        donor_energy=donor_genes.share_threshold + rng.randint(3, 6),
        seed=seed,
        tick=tick,
    )
    donor = safe.organisms[ids["donor"]]
    child = safe.organisms[ids["child"]]
    share = Action(ActionKind.SHARE, target_id=child.organism_id, amount=1)
    records = [
        _energy_record(
            safe.observe(donor),
            share,
            "rescue_share_safe",
            scene_id,
            target,
            True,
            trajectory_step=0,
        )
    ]
    _apply_turn(safe, donor, share)
    donor_after_share = safe.observe(donor)
    no_repeat = _local_control_action(donor_after_share)
    records.append(
        _energy_record(
            donor_after_share,
            no_repeat,
            "rescue_share_not_needed",
            scene_id,
            target,
            True,
            trajectory_step=1,
        )
    )
    child_observation = safe.observe(child)
    child_move = _required_move(child_observation, target)
    records.append(
        _energy_record(
            child_observation,
            child_move,
            "rescue_child_move",
            scene_id,
            target,
            True,
            trajectory_step=1,
        )
    )
    _apply_turn(safe, child, child_move)
    records.append(
        _energy_record(
            safe.observe(donor),
            Action(ActionKind.REST),
            "rescue_share_not_needed_at_food",
            scene_id,
            target,
            True,
            trajectory_step=2,
        )
    )
    records.append(
        _energy_record(
            safe.observe(child),
            Action(ActionKind.FORAGE),
            "rescue_child_forage",
            scene_id,
            target,
            True,
            trajectory_step=2,
        )
    )

    unsafe, unsafe_ids = _build_rescue_world(
        world_config,
        actor,
        target,
        bystanders,
        identity,
        donor_genes,
        child_energy=1,
        donor_energy=max(2, donor_genes.share_threshold - 1),
        seed=seed + 1,
        tick=tick,
    )
    unsafe_donor = unsafe.organisms[unsafe_ids["donor"]]
    unsafe_child = unsafe.organisms[unsafe_ids["child"]]
    records.extend(
        [
            _energy_record(
                unsafe.observe(unsafe_donor),
                _local_control_action(unsafe.observe(unsafe_donor)),
                "rescue_share_unsafe",
                scene_id,
                target,
                True,
                trajectory_step=0,
            ),
            _energy_record(
                unsafe.observe(unsafe_child),
                Action(ActionKind.REST),
                "unrescued_child_wait",
                scene_id,
                target,
                False,
                trajectory_step=0,
            ),
        ]
    )
    return records


def _build_energy_world(
    world_config: WorldConfig,
    actor: Position,
    target: Position,
    bystanders: tuple[Position, Position],
    identity: dict[str, str],
    genes: Genes,
    *,
    energy: int,
    seed: int,
    tick: int,
    memory: list[str],
) -> tuple[World, str]:
    config = replace(
        world_config,
        founders=3,
        initial_food=0,
        food_regrowth=0,
        max_population=max(4, world_config.max_population),
    )
    world = World(config, seed=seed, initialize=False)
    world.tick = tick
    organism = world.spawn_founder(
        actor,
        identity["actor_lineage"],
        energy=energy,
        genes=genes,
        organism_id=identity["actor_id"],
    )
    organism.age = min(tick, 1 if seed % 2 else rng_age(seed, tick))
    for item in memory:
        organism.remember(item, world.config.memory_limit)
    for index, position in enumerate(bystanders, start=1):
        other = world.spawn_founder(
            position,
            identity[f"bystander_{index}_lineage"],
            energy=8 + index,
            genes=Genes(),
            organism_id=identity[f"bystander_{index}_id"],
        )
        other.age = min(tick, (seed + index * 7) % (tick + 1))
    world.add_food(target, 3)
    return world, organism.organism_id


def _build_rescue_world(
    world_config: WorldConfig,
    child_position: Position,
    target: Position,
    bystanders: tuple[Position, Position],
    identity: dict[str, str],
    donor_genes: Genes,
    *,
    child_energy: int,
    donor_energy: int,
    seed: int,
    tick: int,
) -> tuple[World, dict[str, str]]:
    config = replace(
        world_config,
        founders=3,
        initial_food=0,
        food_regrowth=0,
        max_population=max(4, world_config.max_population),
    )
    world = World(config, seed=seed, initialize=False)
    world.tick = tick
    child = world.spawn_founder(
        child_position,
        identity["actor_lineage"],
        energy=child_energy,
        genes=Genes(),
        organism_id=identity["actor_id"],
    )
    child.age = min(5, tick)
    donor = world.spawn_founder(
        bystanders[0],
        identity["bystander_1_lineage"],
        energy=donor_energy,
        genes=donor_genes,
        organism_id=identity["bystander_1_id"],
    )
    donor.age = min(20, tick)
    distractor = world.spawn_founder(
        bystanders[1],
        identity["bystander_2_lineage"],
        energy=9,
        genes=Genes(),
        organism_id=identity["bystander_2_id"],
    )
    distractor.age = min(12, tick)
    world.add_food(target, 3)
    return world, {
        "child": child.organism_id,
        "donor": donor.organism_id,
        "distractor": distractor.organism_id,
    }


def _energy_record(
    observation: Observation,
    action: Action,
    case: str,
    scene_id: str,
    target: Position,
    reachable: bool,
    *,
    trajectory_step: int,
) -> dict[str, Any]:
    distance = _manhattan(observation.position, target)
    record = training_record(observation, action)
    record.update(
        {
            "curriculum": "v4.2",
            "energy_case": case,
            "scene_id": scene_id,
            "trajectory_step": trajectory_step,
            "target_food": list(target),
            "target_distance": distance,
            "energy_budget_required": distance + 1,
            "reachable_before_exhaustion": reachable,
        }
    )
    return record


def _apply_turn(world: World, organism: Any, action: Action) -> None:
    valid, result = world.apply_action(organism, action)
    if not valid:
        raise RuntimeError(f"V4.2 trajectory action failed: {result}")
    organism.age += 1
    organism.energy -= world.config.metabolism
    world.tick += 1
    if organism.energy <= 0:
        raise RuntimeError("V4.2 labelled trajectory killed its actor")


def _required_move(observation: Observation, target: Position) -> Action:
    move = _move_toward(target, observation.open_neighbors)
    if move is None or _manhattan(move, target) >= _manhattan(
        observation.position, target
    ):
        raise RuntimeError("V4.2 trajectory has no open step toward food")
    return Action(ActionKind.MOVE, target_position=move)


def _adjacent_target(
    actor: Position,
    occupied: tuple[Position, Position],
    world_config: WorldConfig,
) -> Position:
    x, y = actor
    candidates = [
        candidate
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if 0 <= candidate[0] < world_config.width
        and 0 <= candidate[1] < world_config.height
        and candidate not in occupied
    ]
    if not candidates:
        raise RuntimeError("could not place adjacent V4.2 food")
    return min(candidates)


def _distance_two_target(
    actor: Position,
    occupied: tuple[Position, Position],
    world_config: WorldConfig,
    rng: Random,
) -> Position:
    candidates: list[Position] = []
    for x in range(world_config.width):
        for y in range(world_config.height):
            target = (x, y)
            if target in occupied or _manhattan(actor, target) != 2:
                continue
            closer_open = [
                neighbor
                for neighbor in _neighbors(actor, world_config)
                if neighbor not in occupied
                and _manhattan(neighbor, target) == 1
            ]
            if closer_open:
                candidates.append(target)
    if not candidates:
        raise RuntimeError("could not place distance-two V4.2 food")
    return rng.choice(candidates)


def _neighbors(position: Position, world_config: WorldConfig) -> list[Position]:
    x, y = position
    return [
        candidate
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if 0 <= candidate[0] < world_config.width
        and 0 <= candidate[1] < world_config.height
    ]


def _failure_memory(
    rng: Random,
    identity: dict[str, str],
    target: Position,
) -> list[str]:
    options = (
        [],
        [f"{identity['bystander_1_id']} signalled: food at [{target[0]},{target[1]}]"],
        [f"{identity['bystander_1_id']} shared 1 energy"],
        [
            f"{identity['bystander_2_id']} signalled: food at [{target[0]},{target[1]}]",
            f"{identity['bystander_1_id']} shared 1 energy",
        ],
    )
    return list(rng.choice(options))


def rng_age(seed: int, tick: int) -> int:
    return (seed * 17 + 11) % (tick + 1)

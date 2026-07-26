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
    _trajectory_move,
)
from .domain import Action, ActionKind, Genes, Observation, Position
from .prompting import action_is_observation_legal, training_record
from .world import World, WorldConfig


V41_RICH_COMMUNICATION_CASES = (
    "rich_sender_useful",
    "rich_sender_redundant",
    "rich_sender_current_food",
    "rich_recipient_informed",
    "rich_recipient_control",
)
V41_REACHABILITY_CASES = (
    "food_reachable",
    "food_unreachable",
    "food_current",
    "rescue_share_safe",
    "rescue_share_unsafe",
    "rescued_child_reachable",
    "unrescued_child_wait",
)


def audit_v41_datasets(
    training_path: str | Path,
    held_out_paths: list[str | Path],
) -> dict[str, object]:
    """Audit SFT legality, neutral identity shape, and held-out overlap."""

    training_records = _read_sft_records(training_path)
    held_out = {
        str(Path(path).resolve()): _read_sft_records(path)
        for path in held_out_paths
    }
    training_fingerprints = {
        _semantic_fingerprint(record) for record in training_records
    }
    training_audit = _audit_records(training_records)
    held_out_audits = {
        path: _audit_records(records) for path, records in held_out.items()
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
            and all(_audit_passes(audit) for audit in held_out_audits.values())
            and not any(overlaps.values())
        ),
    }


def generate_v41_sft_dataset(
    output_path: str | Path,
    world_config: WorldConfig,
    rich_scenes: int,
    reachability_scenes: int,
    seed: int,
    base_dataset_path: str | Path | None = None,
    *,
    include_trajectories: bool = True,
) -> dict[str, object]:
    """Extend V4 using the same SFT format with rich and reachability corrections."""

    if rich_scenes < 0 or reachability_scenes < 0:
        raise ValueError("V4.1 scene counts cannot be negative")
    if (
        rich_scenes == 0
        and reachability_scenes == 0
        and base_dataset_path is None
    ):
        raise ValueError("V4.1 requires generated scenes or a base dataset")
    if world_config.perception_radius < 2:
        raise ValueError("V4.1 scenes require perception radius of at least two")

    source_records = (
        _read_sft_records(base_dataset_path)
        if base_dataset_path is not None
        else []
    )
    records = [_normalize_record(record) for record in source_records]
    fingerprints = {_record_fingerprint(record) for record in records}
    if len(fingerprints) != len(records):
        raise ValueError("V4.1 base dataset contains duplicate prompt/completion records")

    action_counts: Counter[str] = Counter(
        _record_action_kind(record) for record in records
    )
    communication_counts: Counter[str] = Counter()
    reachability_counts: Counter[str] = Counter()
    duplicate_records_skipped = 0
    rng = Random(seed)

    for scene_number in range(rich_scenes):
        scene_records = _rich_communication_records(
            world_config,
            rng,
            seed=seed,
            scene_number=scene_number,
            include_trajectory=include_trajectories,
        )
        for record in scene_records:
            fingerprint = _record_fingerprint(record)
            if fingerprint in fingerprints:
                duplicate_records_skipped += 1
                continue
            fingerprints.add(fingerprint)
            records.append(record)
            action_counts[_record_action_kind(record)] += 1
            communication_counts[str(record["communication_case"])] += 1

    for scene_number in range(reachability_scenes):
        scene_records = _reachability_records(
            world_config,
            rng,
            seed=seed,
            scene_number=scene_number,
        )
        for record in scene_records:
            fingerprint = _record_fingerprint(record)
            if fingerprint in fingerprints:
                duplicate_records_skipped += 1
                continue
            fingerprints.add(fingerprint)
            records.append(record)
            action_counts[_record_action_kind(record)] += 1
            reachability_counts[str(record["survival_case"])] += 1

    Random(seed).shuffle(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, object] = {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "curriculum": "v4.1-rich-reachability",
        "seed": seed,
        "base_dataset": (
            str(Path(base_dataset_path).resolve())
            if base_dataset_path is not None
            else None
        ),
        "base_records": len(source_records),
        "base_prompt_records_normalized": len(source_records),
        "rich_scenes": rich_scenes,
        "reachability_scenes": reachability_scenes,
        "include_trajectories": include_trajectories,
        "communication_case_counts": dict(sorted(communication_counts.items())),
        "reachability_case_counts": dict(sorted(reachability_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_records_skipped": duplicate_records_skipped,
        "unique_prompt_completion_records": len(fingerprints),
        "records": len(records),
    }
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _normalize_record(source: dict[str, Any]) -> dict[str, Any]:
    observation = _observation_from_record(source)
    action = Action.from_payload(json.loads(source["completion"][0]["content"]))
    record = training_record(observation, action)
    record.update(
        {
            key: value
            for key, value in source.items()
            if key not in {"prompt", "completion"}
        }
    )
    return record


def _audit_records(records: list[dict[str, Any]]) -> dict[str, int]:
    fingerprints: set[str] = set()
    duplicates = 0
    illegal = 0
    role_shaped_identities = 0
    age_tick_inconsistencies = 0
    reachability_annotation_errors = 0
    for record in records:
        fingerprint = _semantic_fingerprint(record)
        duplicates += int(fingerprint in fingerprints)
        fingerprints.add(fingerprint)
        observation = _observation_from_record(record)
        action = Action.from_payload(json.loads(record["completion"][0]["content"]))
        legal, _ = action_is_observation_legal(action, observation)
        illegal += int(not legal)
        identities = [
            str(observation["self"]["id"]),
            str(observation["self"]["lineage"]),
        ]
        for agent in observation.get("visible_agents", []):
            identities.extend((str(agent["id"]), str(agent["lineage"])))
        role_shaped_identities += int(
            any(
                value.startswith(
                    ("organism-s", "organism-r", "lineage-s", "lineage-r")
                )
                for value in identities
            )
        )
        age_tick_inconsistencies += int(
            int(observation["self"]["age"]) > int(observation["tick"])
        )
        case = record.get("survival_case")
        if case in {
            "food_reachable",
            "food_unreachable",
            "rescued_child_reachable",
            "unrescued_child_wait",
        }:
            target = tuple(record["target_food"])
            distance = _manhattan(tuple(observation["self"]["position"]), target)
            expected = int(observation["self"]["energy"]) > distance
            reachability_annotation_errors += int(
                bool(record["reachable_before_exhaustion"]) != expected
            )
    return {
        "records": len(records),
        "unique_prompt_completion_records": len(fingerprints),
        "duplicate_prompt_completion_records": duplicates,
        "illegal_labels": illegal,
        "role_shaped_identity_records": role_shaped_identities,
        "age_tick_inconsistencies": age_tick_inconsistencies,
        "reachability_annotation_errors": reachability_annotation_errors,
    }


def _audit_passes(audit: dict[str, int]) -> bool:
    return all(
        audit[key] == 0
        for key in (
            "duplicate_prompt_completion_records",
            "illegal_labels",
            "role_shaped_identity_records",
            "age_tick_inconsistencies",
            "reachability_annotation_errors",
        )
    )


def _semantic_fingerprint(record: dict[str, Any]) -> str:
    payload = {
        "observation": _observation_from_record(record),
        "completion": record["completion"],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _rich_communication_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    seed: int,
    scene_number: int,
    include_trajectory: bool,
) -> list[dict[str, Any]]:
    layout = _sample_rich_layout(world_config, rng)
    identity = _sample_rich_identity(rng, seed, scene_number)
    state = _sample_rich_state(rng, seed, scene_number, identity, layout["target"])

    useful_world, useful_ids = _build_rich_world(
        world_config,
        layout,
        identity,
        state,
        recipient_position=layout["useful_recipient"],
    )
    redundant_world, redundant_ids = _build_rich_world(
        world_config,
        layout,
        identity,
        state,
        recipient_position=layout["redundant_recipient"],
    )
    current_world, current_ids = _build_rich_world(
        world_config,
        layout,
        identity,
        state,
        recipient_position=layout["useful_recipient"],
        sender_current_food=True,
    )
    if useful_ids != redundant_ids or useful_ids != current_ids:
        raise RuntimeError("V4.1 paired rich worlds changed neutral identities")

    sender_id = useful_ids["sender"]
    recipient_id = useful_ids["recipient"]
    target = layout["target"]
    sender_useful = useful_world.observe(useful_world.organisms[sender_id])
    recipient_control = useful_world.observe(useful_world.organisms[recipient_id])
    sender_redundant = redundant_world.observe(
        redundant_world.organisms[sender_id]
    )
    sender_current = current_world.observe(current_world.organisms[sender_id])

    if len(sender_useful.visible_agents) != 3 or len(sender_useful.visible_food) != 2:
        raise RuntimeError("V4.1 useful sender lacks rich multi-agent/multi-food context")
    if recipient_control.current_food or recipient_control.visible_food:
        raise RuntimeError("V4.1 useful recipient already sees food")
    for agent in sender_redundant.visible_agents:
        peer = redundant_world.organisms[str(agent["id"])]
        peer_observation = redundant_world.observe(peer)
        if not peer_observation.current_food and not peer_observation.visible_food:
            raise RuntimeError("V4.1 redundant sender still has an uninformed peer")

    redundant_move = _move_toward(target, sender_redundant.open_neighbors)
    if redundant_move is None:
        raise RuntimeError("V4.1 redundant sender cannot approach target food")
    signal = Action(
        ActionKind.SIGNAL,
        message=f"food at [{target[0]},{target[1]}]",
    )
    valid, _ = useful_world.apply_action(useful_world.organisms[sender_id], signal)
    if not valid:
        raise RuntimeError("V4.1 rich signal was unexpectedly invalid")
    recipient_informed = useful_world.observe(useful_world.organisms[recipient_id])
    informed_move = _move_toward(target, recipient_informed.open_neighbors)
    if informed_move is None:
        raise RuntimeError("V4.1 rich recipient cannot approach signalled food")

    scene_id = f"v41-rich-{seed}-{scene_number:05d}"
    records = [
        _communication_record(
            sender_useful,
            signal,
            "rich_sender_useful",
            f"{scene_id}-sender",
            scene_id,
            target,
        ),
        _communication_record(
            sender_redundant,
            Action(ActionKind.MOVE, target_position=redundant_move),
            "rich_sender_redundant",
            f"{scene_id}-sender",
            scene_id,
            target,
        ),
        _communication_record(
            sender_current,
            Action(ActionKind.FORAGE),
            "rich_sender_current_food",
            f"{scene_id}-priority",
            scene_id,
            target,
        ),
        _communication_record(
            recipient_informed,
            Action(ActionKind.MOVE, target_position=informed_move),
            "rich_recipient_informed",
            f"{scene_id}-recipient",
            scene_id,
            target,
        ),
        _communication_record(
            recipient_control,
            _local_control_action(recipient_control),
            "rich_recipient_control",
            f"{scene_id}-recipient",
            scene_id,
            target,
        ),
    ]
    if include_trajectory:
        records.extend(
            _rich_trajectory_records(
                world_config,
                layout,
                identity,
                state,
                scene_id,
            )
        )
    return records


def _rich_trajectory_records(
    world_config: WorldConfig,
    layout: dict[str, Position],
    identity: dict[str, str],
    state: dict[str, Any],
    scene_id: str,
) -> list[dict[str, Any]]:
    trajectory_state = dict(state)
    trajectory_state["recipient_energy"] = max(
        int(state["recipient_energy"]),
        world_config.perception_radius * 4 + 20,
    )
    world, ids = _build_rich_world(
        world_config,
        layout,
        identity,
        trajectory_state,
        recipient_position=layout["useful_recipient"],
    )
    policy = _RichTrajectoryPolicy(
        sender_id=ids["sender"],
        recipient_id=ids["recipient"],
        target=layout["target"],
    )
    world.run(policy, steps=world.config.perception_radius * 4 + 18)
    records: list[dict[str, Any]] = []
    step = 0
    for event in world.events:
        if event.kind != "decision" or event.actor_id != ids["recipient"]:
            continue
        observation = event.data["observation"]
        expected_memory = (
            f"food at [{layout['target'][0]},{layout['target'][1]}]"
        )
        if not any(expected_memory in item for item in observation["memory"]):
            continue
        action = Action.from_payload(event.data["action"])
        if not observation["visible_food"] and not observation["current_food"]:
            continue
        record = _communication_record(
            observation,
            action,
            "rich_recipient_trajectory",
            f"{scene_id}-trajectory-{step:02d}",
            scene_id,
            layout["target"],
        )
        record["trajectory_step"] = step
        records.append(record)
        step += 1
        if action.kind is ActionKind.FORAGE:
            break
    if not records or _record_action_kind(records[-1]) != ActionKind.FORAGE.value:
        return []
    return records


def _sample_rich_layout(
    world_config: WorldConfig,
    rng: Random,
) -> dict[str, Position]:
    radius = world_config.perception_radius
    cells = [
        (x, y)
        for x in range(world_config.width)
        for y in range(world_config.height)
    ]
    for _ in range(20_000):
        sender = rng.choice(cells)
        nearby = [
            cell
            for cell in cells
            if cell != sender and _manhattan(sender, cell) <= radius
        ]
        useful = rng.choice(nearby)
        targets = [
            cell
            for cell in nearby
            if cell != useful
            and _manhattan(sender, cell) >= 2
            and _manhattan(useful, cell) > radius
        ]
        if not targets:
            continue
        target = rng.choice(targets)
        extras = [
            cell
            for cell in nearby
            if cell not in {useful, target}
            and _manhattan(useful, cell) > radius
            and _manhattan(sender, cell) > _manhattan(sender, target)
        ]
        if not extras:
            continue
        extra = rng.choice(extras)
        redundant = [
            cell
            for cell in nearby
            if cell not in {target, extra, useful}
            and _manhattan(cell, target) <= radius
            and _manhattan(cell, extra) <= radius
        ]
        if not redundant:
            continue
        redundant_recipient = rng.choice(redundant)
        distractor_candidates = [
            cell
            for cell in nearby
            if cell not in {
                useful,
                redundant_recipient,
                target,
                extra,
            }
            and (
                _manhattan(cell, target) <= radius
                or _manhattan(cell, extra) <= radius
            )
        ]
        if len(distractor_candidates) < 2:
            continue
        rng.shuffle(distractor_candidates)
        layout = {
            "sender": sender,
            "useful_recipient": useful,
            "redundant_recipient": redundant_recipient,
            "target": target,
            "extra_food": extra,
            "distractor_one": distractor_candidates[0],
            "distractor_two": distractor_candidates[1],
        }
        blockers = {
            sender,
            layout["distractor_one"],
            layout["distractor_two"],
        }
        occupied = {
            sender,
            useful,
            layout["distractor_one"],
            layout["distractor_two"],
        }
        redundant_occupied = {
            sender,
            redundant_recipient,
            layout["distractor_one"],
            layout["distractor_two"],
        }
        if any(
            _manhattan(neighbor, target) < _manhattan(sender, target)
            and neighbor not in occupied
            for neighbor in _neighbors(sender, world_config)
        ) and any(
            _manhattan(neighbor, target) < _manhattan(sender, target)
            and neighbor not in redundant_occupied
            for neighbor in _neighbors(sender, world_config)
        ) and _greedy_path_reaches(
            useful,
            target,
            blockers,
            world_config,
            steps=radius * 3 + 4,
        ):
            return layout
    raise RuntimeError("could not construct a V4.1 rich communication layout")


def _sample_rich_identity(
    rng: Random,
    seed: int,
    scene_number: int,
) -> dict[str, str]:
    roles = ("sender", "recipient", "distractor_one", "distractor_two")
    prefix = abs(seed) % 1_000_000
    ids = [
        f"organism-{prefix:06d}-{scene_number * 4 + offset:07d}"
        for offset in range(1, 5)
    ]
    lineages = [
        f"lineage-{(prefix * 37 + scene_number * 4 + offset) % 10_000_000:07d}"
        for offset in range(1, 5)
    ]
    rng.shuffle(ids)
    rng.shuffle(lineages)
    result: dict[str, str] = {}
    for role, organism_id, lineage_id in zip(roles, ids, lineages, strict=True):
        result[f"{role}_id"] = organism_id
        result[f"{role}_lineage"] = lineage_id
    return result


def _sample_rich_state(
    rng: Random,
    seed: int,
    scene_number: int,
    identity: dict[str, str],
    target: Position,
) -> dict[str, Any]:
    roles = ("sender", "recipient", "distractor_one", "distractor_two")
    tick = rng.randint(0, 160)
    state: dict[str, Any] = {
        "world_seed": seed * 100_000 + scene_number,
        "tick": tick,
        "spawn_order": list(roles),
    }
    rng.shuffle(state["spawn_order"])
    for index, role in enumerate(roles):
        state[f"{role}_age"] = rng.randint(0, tick)
        state[f"{role}_energy"] = rng.randint(8, 15)
        state[f"{role}_genes"] = _sample_genes(rng)
        other_role = roles[(index + 1) % len(roles)]
        memory: list[str] = []
        if rng.randrange(3) == 1:
            memory.append(f"{identity[f'{other_role}_id']} shared 1 energy")
        elif rng.randrange(3) == 2:
            wrong = (
                target[0] - 1 if target[0] > 0 else target[0] + 1,
                target[1] - 1 if target[1] > 0 else target[1] + 1,
            )
            memory.append(
                f"{identity[f'{other_role}_id']} signalled: "
                f"food at [{wrong[0]},{wrong[1]}]"
            )
        state[f"{role}_memory"] = memory
    return state


def _build_rich_world(
    world_config: WorldConfig,
    layout: dict[str, Position],
    identity: dict[str, str],
    state: dict[str, Any],
    *,
    recipient_position: Position,
    sender_current_food: bool = False,
) -> tuple[World, dict[str, str]]:
    config = replace(
        world_config,
        founders=4,
        initial_food=0,
        food_regrowth=0,
        max_food=max(8, world_config.max_food),
        max_population=max(6, world_config.max_population),
        signal_delivery="normal",
        signal_cost=0,
    )
    world = World(config, seed=int(state["world_seed"]), initialize=False)
    world.tick = int(state["tick"])
    positions = {
        "sender": layout["sender"],
        "recipient": recipient_position,
        "distractor_one": layout["distractor_one"],
        "distractor_two": layout["distractor_two"],
    }
    ids: dict[str, str] = {}
    for role in state["spawn_order"]:
        organism = world.spawn_founder(
            position=positions[role],
            lineage_id=identity[f"{role}_lineage"],
            energy=int(state[f"{role}_energy"]),
            genes=state[f"{role}_genes"],
            organism_id=identity[f"{role}_id"],
        )
        organism.age = int(state[f"{role}_age"])
        for item in state[f"{role}_memory"]:
            organism.remember(item, world.config.memory_limit)
        ids[role] = organism.organism_id
    world.add_food(layout["target"], 3)
    world.add_food(layout["extra_food"], 1)
    if sender_current_food:
        world.add_food(layout["sender"], 2)
    return world, ids


def _reachability_records(
    world_config: WorldConfig,
    rng: Random,
    *,
    seed: int,
    scene_number: int,
) -> list[dict[str, Any]]:
    actor, target, bystander_positions = _sample_reachability_layout(world_config, rng)
    distance = _manhattan(actor, target)
    identity = _sample_reachability_identity(rng, seed, scene_number)
    genes = _sample_genes(rng)
    scene_id = f"v41-reach-{seed}-{scene_number:05d}"

    reachable_world, reachable_id = _build_reachability_world(
        world_config,
        actor,
        target,
        bystander_positions,
        identity,
        genes,
        actor_energy=distance + rng.randint(1, 3),
        seed=seed * 200_000 + scene_number,
        tick=rng.randint(0, 160),
    )
    unreachable_world, unreachable_id = _build_reachability_world(
        world_config,
        actor,
        target,
        bystander_positions,
        identity,
        genes,
        actor_energy=rng.randint(1, distance),
        seed=seed * 200_000 + scene_number,
        tick=reachable_world.tick,
    )
    current_world, current_id = _build_reachability_world(
        world_config,
        actor,
        target,
        bystander_positions,
        identity,
        genes,
        actor_energy=1,
        seed=seed * 200_000 + scene_number,
        tick=reachable_world.tick,
        current_food=True,
    )
    reachable_observation = reachable_world.observe(
        reachable_world.organisms[reachable_id]
    )
    reachable_move = _move_toward(target, reachable_observation.open_neighbors)
    if reachable_move is None:
        raise RuntimeError("V4.1 reachable organism has no path step")

    child = actor
    rescue_target = _rescue_target(child, bystander_positions, world_config)
    donor = bystander_positions[0]
    distractor = bystander_positions[1]
    safe_world, rescue_ids = _build_rescue_world(
        world_config,
        child,
        donor,
        distractor,
        rescue_target,
        identity,
        genes,
        donor_energy=genes.share_threshold + rng.randint(3, 6),
        seed=seed * 300_000 + scene_number,
        tick=reachable_world.tick,
    )
    unsafe_world, unsafe_ids = _build_rescue_world(
        world_config,
        child,
        donor,
        distractor,
        rescue_target,
        identity,
        genes,
        donor_energy=max(3, genes.share_threshold - rng.randint(1, 3)),
        seed=seed * 300_000 + scene_number,
        tick=reachable_world.tick,
    )
    safe_share = Action(
        ActionKind.SHARE,
        target_id=rescue_ids["child"],
        amount=1,
    )
    donor_observation = safe_world.observe(safe_world.organisms[rescue_ids["donor"]])
    valid, _ = safe_world.apply_action(
        safe_world.organisms[rescue_ids["donor"]],
        safe_share,
    )
    if not valid:
        raise RuntimeError("V4.1 rescue share was unexpectedly invalid")
    rescued_child = safe_world.observe(safe_world.organisms[rescue_ids["child"]])
    rescued_move = _move_toward(rescue_target, rescued_child.open_neighbors)
    if rescued_move is None or rescued_child.energy <= 1:
        raise RuntimeError("V4.1 rescue did not make target reachable")
    unsafe_donor_observation = unsafe_world.observe(
        unsafe_world.organisms[unsafe_ids["donor"]]
    )
    unsafe_move = _move_toward(
        rescue_target,
        unsafe_donor_observation.open_neighbors,
    )
    unsafe_action = (
        Action(ActionKind.MOVE, target_position=unsafe_move)
        if unsafe_move is not None
        else Action(ActionKind.REST)
    )

    return [
        _survival_record(
            reachable_observation,
            Action(ActionKind.MOVE, target_position=reachable_move),
            "food_reachable",
            scene_id,
            target,
            True,
        ),
        _survival_record(
            unreachable_world.observe(unreachable_world.organisms[unreachable_id]),
            Action(ActionKind.REST),
            "food_unreachable",
            scene_id,
            target,
            False,
        ),
        _survival_record(
            current_world.observe(current_world.organisms[current_id]),
            Action(ActionKind.FORAGE),
            "food_current",
            scene_id,
            actor,
            True,
        ),
        _survival_record(
            donor_observation,
            safe_share,
            "rescue_share_safe",
            scene_id,
            rescue_target,
            True,
        ),
        _survival_record(
            unsafe_donor_observation,
            unsafe_action,
            "rescue_share_unsafe",
            scene_id,
            rescue_target,
            True,
        ),
        _survival_record(
            rescued_child,
            Action(ActionKind.MOVE, target_position=rescued_move),
            "rescued_child_reachable",
            scene_id,
            rescue_target,
            True,
        ),
        _survival_record(
            unsafe_world.observe(unsafe_world.organisms[unsafe_ids["child"]]),
            Action(ActionKind.REST),
            "unrescued_child_wait",
            scene_id,
            rescue_target,
            False,
        ),
    ]


def _sample_reachability_layout(
    world_config: WorldConfig,
    rng: Random,
) -> tuple[Position, Position, tuple[Position, Position]]:
    radius = world_config.perception_radius
    for _ in range(10_000):
        actor = (
            rng.randrange(1, world_config.width - 1),
            rng.randrange(1, world_config.height - 1),
        )
        targets = [
            position
            for position in _positions_within(actor, radius, world_config)
            if 2 <= _manhattan(actor, position) <= radius
        ]
        if not targets:
            continue
        target = rng.choice(targets)
        neighbors = _neighbors(actor, world_config)
        donor_candidates = [
            position
            for position in neighbors
            if position != target
        ]
        if not donor_candidates:
            continue
        donor = rng.choice(donor_candidates)
        distractor_candidates = [
            position
            for position in _positions_within(actor, radius, world_config)
            if position not in {actor, target, donor}
            and _manhattan(position, actor) >= 2
        ]
        if distractor_candidates:
            return actor, target, (donor, rng.choice(distractor_candidates))
    raise RuntimeError("could not construct a V4.1 reachability layout")


def _build_reachability_world(
    world_config: WorldConfig,
    actor_position: Position,
    target: Position,
    bystander_positions: tuple[Position, Position],
    identity: dict[str, str],
    actor_genes: Genes,
    *,
    actor_energy: int,
    seed: int,
    tick: int,
    current_food: bool = False,
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
    actor = world.spawn_founder(
        actor_position,
        identity["actor_lineage"],
        energy=actor_energy,
        genes=actor_genes,
        organism_id=identity["actor_id"],
    )
    for index, position in enumerate(bystander_positions, start=1):
        world.spawn_founder(
            position,
            identity[f"bystander_{index}_lineage"],
            energy=8 + index,
            genes=Genes(),
            organism_id=identity[f"bystander_{index}_id"],
        )
    world.add_food(actor_position if current_food else target, 3)
    return world, actor.organism_id


def _build_rescue_world(
    world_config: WorldConfig,
    child_position: Position,
    donor_position: Position,
    distractor_position: Position,
    target: Position,
    identity: dict[str, str],
    donor_genes: Genes,
    *,
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
        energy=1,
        genes=Genes(),
        organism_id=identity["actor_id"],
    )
    donor = world.spawn_founder(
        donor_position,
        identity["bystander_1_lineage"],
        energy=donor_energy,
        genes=donor_genes,
        organism_id=identity["bystander_1_id"],
    )
    distractor = world.spawn_founder(
        distractor_position,
        identity["bystander_2_lineage"],
        energy=9,
        genes=Genes(),
        organism_id=identity["bystander_2_id"],
    )
    world.add_food(target, 3)
    return world, {
        "child": child.organism_id,
        "donor": donor.organism_id,
        "distractor": distractor.organism_id,
    }


def _sample_reachability_identity(
    rng: Random,
    seed: int,
    scene_number: int,
) -> dict[str, str]:
    roles = ("actor", "bystander_1", "bystander_2")
    prefix = (abs(seed) + 700_000) % 1_000_000
    ids = [
        f"organism-{prefix:06d}-{scene_number * 3 + offset:07d}"
        for offset in range(1, 4)
    ]
    lineages = [
        f"lineage-{(prefix * 41 + scene_number * 3 + offset) % 10_000_000:07d}"
        for offset in range(1, 4)
    ]
    rng.shuffle(ids)
    rng.shuffle(lineages)
    result: dict[str, str] = {}
    for role, organism_id, lineage_id in zip(roles, ids, lineages, strict=True):
        result[f"{role}_id"] = organism_id
        result[f"{role}_lineage"] = lineage_id
    return result


def _rescue_target(
    child: Position,
    occupied: tuple[Position, Position],
    world_config: WorldConfig,
) -> Position:
    candidates = [
        position
        for position in _neighbors(child, world_config)
        if position not in occupied
    ]
    if not candidates:
        raise RuntimeError("could not place adjacent V4.1 rescue food")
    return min(candidates)


def _positions_within(
    origin: Position,
    radius: int,
    world_config: WorldConfig,
) -> list[Position]:
    return [
        (x, y)
        for x in range(max(0, origin[0] - radius), min(world_config.width, origin[0] + radius + 1))
        for y in range(max(0, origin[1] - radius), min(world_config.height, origin[1] + radius + 1))
        if _manhattan(origin, (x, y)) <= radius
    ]


def _neighbors(position: Position, world_config: WorldConfig) -> list[Position]:
    x, y = position
    return [
        candidate
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if 0 <= candidate[0] < world_config.width
        and 0 <= candidate[1] < world_config.height
    ]


def _greedy_path_reaches(
    start: Position,
    target: Position,
    blocked: set[Position],
    world_config: WorldConfig,
    *,
    steps: int,
) -> bool:
    position = start
    for _ in range(steps):
        if position == target:
            return True
        candidates = [
            candidate
            for candidate in _neighbors(position, world_config)
            if candidate not in blocked
        ]
        if not candidates:
            return False
        position = min(
            candidates,
            key=lambda candidate: (_manhattan(candidate, target), candidate),
        )
    return position == target


def _communication_record(
    observation: Observation | dict[str, Any],
    action: Action,
    case: str,
    pair_id: str,
    scene_id: str,
    target: Position,
) -> dict[str, Any]:
    record = training_record(observation, action)
    visible_agents = (
        observation.visible_agents
        if isinstance(observation, Observation)
        else observation.get("visible_agents", [])
    )
    visible_food = (
        observation.visible_food
        if isinstance(observation, Observation)
        else observation.get("visible_food", [])
    )
    record.update(
        {
            "curriculum": "v4.1",
            "communication_case": case,
            "pair_id": pair_id,
            "scene_id": scene_id,
            "target_food": list(target),
            "context_agents": len(visible_agents) + 1,
            "visible_food_count": len(visible_food),
        }
    )
    return record


def _survival_record(
    observation: Observation,
    action: Action,
    case: str,
    scene_id: str,
    target: Position,
    reachable: bool,
) -> dict[str, Any]:
    distance = _manhattan(observation.position, target)
    record = training_record(observation, action)
    record.update(
        {
            "curriculum": "v4.1",
            "survival_case": case,
            "scene_id": scene_id,
            "target_food": list(target),
            "target_distance": distance,
            "reachable_before_exhaustion": reachable,
        }
    )
    return record


class _RichTrajectoryPolicy:
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
        self.visited: set[Position] = set()

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
        self.visited.add(observation.position)
        if observation.current_food > 0:
            return Action(ActionKind.FORAGE)
        if any(
            f"food at [{self.target[0]},{self.target[1]}]" in item
            for item in observation.memory
        ):
            move = _trajectory_move(observation, self.target, self.visited)
            if move is not None:
                return Action(ActionKind.MOVE, target_position=move)
        return Action(ActionKind.REST)

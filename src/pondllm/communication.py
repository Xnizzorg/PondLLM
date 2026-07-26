from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable

from .domain import Action, ActionKind, Event, Genes, Observation, Position
from .policies import Policy
from .prompting import strict_json_action
from .world import World, WorldConfig


COMMUNICATION_CONDITIONS = ("normal", "blocked", "corrupted", "costly")
COMMUNICATION_PROFILES = ("matched", "clean", "v4", "v41")


@dataclass(frozen=True, slots=True)
class CommunicationScene:
    seed: int
    profile: str
    initial_tick: int
    sender_id: str
    recipient_id: str
    sender_lineage: str
    recipient_lineage: str
    sender_position: Position
    recipient_position: Position
    target_food: Position
    target_amount: int
    sender_energy: int
    recipient_energy: int
    sender_age: int
    recipient_age: int
    distractor_ids: tuple[str, ...]
    distractor_positions: tuple[Position, ...]
    extra_food: tuple[tuple[int, int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_communication_world(
    seed: int,
    condition: str,
    profile: str = "matched",
) -> tuple[World, CommunicationScene]:
    if condition not in COMMUNICATION_CONDITIONS:
        raise ValueError(f"unknown communication condition: {condition}")
    if profile not in COMMUNICATION_PROFILES:
        raise ValueError(f"unknown communication profile: {profile}")

    distractor_positions: tuple[Position, ...] = ()
    extra_food: tuple[tuple[int, int, int], ...] = ()
    if profile == "v41":
        width = 11
        height = 11
        (
            sender_position,
            recipient_position,
            target_food,
            extra_food_position,
            distractor_positions,
        ) = _v41_scene_positions(
            seed,
            width=width,
            height=height,
            radius=3,
        )
        extra_food = ((extra_food_position[0], extra_food_position[1], 1),)
    elif profile == "v4":
        width = 11
        height = 11
        sender_position, recipient_position, target_food = _v4_scene_positions(
            seed,
            width=width,
            height=height,
            radius=3,
        )
    else:
        width = 7
        height = 7
        sender_position, recipient_position, target_food = _scene_positions(seed)
    delivery = "normal" if condition == "costly" else condition
    signal_cost = 1 if condition == "costly" else 0
    config = WorldConfig(
        width=width,
        height=height,
        founders=4 if profile == "v41" else 2,
        initial_energy=10,
        initial_food=0,
        food_energy=4,
        food_regrowth=0,
        max_food=12,
        metabolism=1,
        move_cost=0,
        rest_gain=1,
        reproduction_cost=8,
        child_energy=4,
        max_population=6 if profile == "v41" else 4,
        memory_limit=12,
        perception_radius=3,
        signal_delivery=delivery,
        signal_cost=signal_cost,
    )
    world = World(config, seed=seed, initialize=False)
    if profile == "matched":
        initial_tick = 2 + seed % 89
    elif profile in {"v4", "v41"}:
        initial_tick = (seed * 17) % 121
    else:
        initial_tick = 0
    world.tick = initial_tick
    sender_energy = 9 + seed % 4
    recipient_energy = (9 if profile in {"v4", "v41"} else 7) + seed % 3
    if profile == "matched":
        sender_id = f"organism-s{seed}-{seed:05d}"
        recipient_id = f"organism-r{seed}-{seed:05d}"
        sender_lineage = f"lineage-s{seed % 17:02d}"
        recipient_lineage = f"lineage-r{seed % 19:02d}"
    elif profile == "clean":
        sender_id = None
        recipient_id = None
        sender_lineage = "lineage-00001"
        recipient_lineage = "lineage-00002"
    elif profile == "v4":
        rng = Random(seed + 91_003)
        organism_ids = [
            f"organism-{seed:06d}-00001",
            f"organism-{seed:06d}-00002",
        ]
        lineages = [
            f"lineage-{(seed * 31 + 1) % 10_000_000:07d}",
            f"lineage-{(seed * 31 + 2) % 10_000_000:07d}",
        ]
        rng.shuffle(organism_ids)
        rng.shuffle(lineages)
        sender_id, recipient_id = organism_ids
        sender_lineage, recipient_lineage = lineages
    else:
        rng = Random(seed + 91_003)
        organism_ids = [
            f"organism-{seed:06d}-{index:05d}" for index in range(1, 5)
        ]
        lineages = [
            f"lineage-{(seed * 31 + index) % 10_000_000:07d}"
            for index in range(1, 5)
        ]
        rng.shuffle(organism_ids)
        rng.shuffle(lineages)
        sender_id, recipient_id = organism_ids[:2]
        sender_lineage, recipient_lineage = lineages[:2]
    sender = world.spawn_founder(
        position=sender_position,
        lineage_id=sender_lineage,
        energy=sender_energy,
        genes=Genes(),
        organism_id=sender_id,
    )
    recipient = world.spawn_founder(
        position=recipient_position,
        lineage_id=recipient_lineage,
        energy=recipient_energy,
        genes=Genes(),
        organism_id=recipient_id,
    )
    distractor_ids: list[str] = []
    if profile == "v41":
        for index, position in enumerate(distractor_positions, start=2):
            distractor = world.spawn_founder(
                position=position,
                lineage_id=lineages[index],
                energy=8 + (seed + index) % 5,
                genes=Genes(),
                organism_id=organism_ids[index],
            )
            distractor_ids.append(distractor.organism_id)
    if profile in {"v4", "v41"}:
        rng = Random(seed + 43_901)
        sender.age = rng.randint(0, initial_tick)
        recipient.age = rng.randint(0, initial_tick)
        for distractor_id in distractor_ids:
            world.organisms[distractor_id].age = rng.randint(0, initial_tick)
    else:
        sender.age = 0
        recipient.age = 0
    if profile == "matched":
        sender.remember(
            f"last forage was {1 + seed % 8} ticks ago",
            world.config.memory_limit,
        )
    elif profile in {"v4", "v41"}:
        if seed % 3 == 0:
            sender.remember(
                f"{recipient.organism_id} shared 1 energy",
                world.config.memory_limit,
            )
        elif seed % 3 == 1:
            recipient.remember(
                f"{sender.organism_id} shared 1 energy",
                world.config.memory_limit,
            )
        recipient.remember(
            f"survived tick {initial_tick - 1}",
            world.config.memory_limit,
        )
    target_amount = 3
    world.add_food(target_food, target_amount)
    for x, y, amount in extra_food:
        world.add_food((x, y), amount)
    scene = CommunicationScene(
        seed=seed,
        profile=profile,
        initial_tick=initial_tick,
        sender_id=sender.organism_id,
        recipient_id=recipient.organism_id,
        sender_lineage=sender.lineage_id,
        recipient_lineage=recipient.lineage_id,
        sender_position=sender_position,
        recipient_position=recipient_position,
        target_food=target_food,
        target_amount=target_amount,
        sender_energy=sender_energy,
        recipient_energy=recipient_energy,
        sender_age=sender.age,
        recipient_age=recipient.age,
        distractor_ids=tuple(distractor_ids),
        distractor_positions=distractor_positions,
        extra_food=extra_food,
    )
    return world, scene


def run_communication_experiment(
    policy: Policy,
    output_dir: str | Path,
    seeds: Iterable[int],
    steps: int = 7,
    profile: str = "matched",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("communication experiment seeds must be unique")
    if profile not in COMMUNICATION_PROFILES:
        raise ValueError(f"unknown communication profile: {profile}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in seed_values:
        expected_scene: dict[str, Any] | None = None
        for condition in COMMUNICATION_CONDITIONS:
            world, scene = create_communication_world(seed, condition, profile=profile)
            scene_payload = scene.to_dict()
            if expected_scene is None:
                expected_scene = scene_payload
            elif scene_payload != expected_scene:
                raise RuntimeError("paired communication worlds do not share an initial scene")

            world.run(
                _PairedCommunicationPolicy(
                    delegate=policy,
                    sender_id=scene.sender_id,
                    sender_action_tick=scene.initial_tick,
                    forced_rest_ids=scene.distractor_ids,
                ),
                steps,
            )
            run_summary = _summarize_world(world, scene, condition)
            run_path = destination / f"seed-{seed:05d}-{condition}"
            run_path.mkdir(parents=True, exist_ok=True)
            world.write_events(run_path / "events.jsonl")
            with (run_path / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(run_summary, handle, indent=2)
            runs.append(run_summary)

    summary = {
        "seeds": seed_values,
        "steps": steps,
        "profile": profile,
        "conditions": list(COMMUNICATION_CONDITIONS),
        "metadata": metadata or {},
        "runs": runs,
        "per_condition": {
            condition: _aggregate_condition(
                [run for run in runs if run["condition"] == condition]
            )
            for condition in COMMUNICATION_CONDITIONS
        },
        "paired_effects": _paired_effects(runs, seed_values),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _summarize_world(
    world: World,
    scene: CommunicationScene,
    condition: str,
) -> dict[str, Any]:
    decisions = [event for event in world.events if event.kind == "decision"]
    forced_sender_decisions = [
        event
        for event in decisions
        if (
            event.actor_id in scene.distractor_ids
            or (
                event.actor_id == scene.sender_id
                and event.tick > scene.initial_tick
            )
        )
    ]
    model_decisions = [
        event for event in decisions if event not in forced_sender_decisions
    ]
    signals = [
        event
        for event in world.events
        if event.kind == "signal" and event.actor_id == scene.sender_id
    ]
    recipient_decisions = [
        event for event in decisions if event.actor_id == scene.recipient_id
    ]
    recipient_actions = Counter(
        str(event.data["action"]["action"]) for event in recipient_decisions
    )
    sender_actions = Counter(
        str(event.data["action"]["action"])
        for event in decisions
        if event.actor_id == scene.sender_id
    )
    delivered = [
        event for event in signals if scene.recipient_id in event.data.get("receivers", [])
    ]
    informed_decisions = [
        event
        for event in recipient_decisions
        if any("signalled:" in item for item in event.data["observation"]["memory"])
    ]
    recipient_forages = [
        event
        for event in world.events
        if event.kind == "forage" and event.actor_id == scene.recipient_id
    ]
    recipient_target_observations = [
        event
        for event in recipient_decisions
        if tuple(event.data["observation"]["self"]["position"]) == scene.target_food
    ]
    recipient = world.organisms[scene.recipient_id]
    sender = world.organisms[scene.sender_id]
    return {
        "seed": scene.seed,
        "condition": condition,
        "scene": scene.to_dict(),
        "decisions": len(decisions),
        "valid_decisions": sum(bool(event.data["valid"]) for event in decisions),
        "forced_sender_decisions": len(forced_sender_decisions),
        "model_decisions": len(model_decisions),
        "valid_model_decisions": sum(
            bool(event.data["valid"]) and "policy_error" not in event.data
            for event in model_decisions
        ),
        "strict_model_outputs": sum(
            strict_json_action(event.data.get("raw_output")) is not None
            for event in model_decisions
        ),
        "model_policy_errors": sum(
            "policy_error" in event.data for event in model_decisions
        ),
        "sender_actions": dict(sorted(sender_actions.items())),
        "recipient_actions": dict(sorted(recipient_actions.items())),
        "sender_signal_count": len(signals),
        "signals_delivered_to_recipient": len(delivered),
        "recipient_informed": bool(informed_decisions),
        "recipient_first_informed_tick": (
            min(event.tick for event in informed_decisions) if informed_decisions else None
        ),
        "recipient_reached_food": (
            recipient.position == scene.target_food or bool(recipient_target_observations)
        ),
        "recipient_first_target_tick": (
            min(event.tick for event in recipient_target_observations)
            if recipient_target_observations
            else None
        ),
        "recipient_forage_count": len(recipient_forages),
        "recipient_first_forage_tick": (
            min(event.tick for event in recipient_forages) if recipient_forages else None
        ),
        "recipient_final_position": list(recipient.position),
        "recipient_final_distance": _manhattan(recipient.position, scene.target_food),
        "recipient_final_energy": recipient.energy,
        "recipient_alive": recipient.alive,
        "sender_final_energy": sender.energy,
        "sender_alive": sender.alive,
        "signal_energy_spent": sum(int(event.data.get("cost", 0)) for event in signals),
        "messages": [_message_payload(event) for event in signals],
        "world": world.summary(),
    }


def _message_payload(event: Event) -> dict[str, Any]:
    return {
        "tick": event.tick,
        "message": event.data.get("message"),
        "delivered_message": event.data.get("delivered_message"),
        "receivers": event.data.get("receivers", []),
        "delivery": event.data.get("delivery"),
        "cost": event.data.get("cost", 0),
    }


def _aggregate_condition(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = sum(int(run["decisions"]) for run in runs)
    model_decisions = sum(int(run["model_decisions"]) for run in runs)
    return {
        "runs": len(runs),
        "valid_action_rate": (
            round(sum(int(run["valid_decisions"]) for run in runs) / decisions, 4)
            if decisions
            else 0.0
        ),
        "model_decisions": model_decisions,
        "model_action_valid_rate": (
            round(
                sum(int(run["valid_model_decisions"]) for run in runs)
                / model_decisions,
                4,
            )
            if model_decisions
            else 0.0
        ),
        "strict_json_rate": (
            round(
                sum(int(run["strict_model_outputs"]) for run in runs)
                / model_decisions,
                4,
            )
            if model_decisions
            else 0.0
        ),
        "model_policy_error_rate": (
            round(
                sum(int(run["model_policy_errors"]) for run in runs)
                / model_decisions,
                4,
            )
            if model_decisions
            else 0.0
        ),
        "sender_signalled_rate": round(
            sum(int(run["sender_signal_count"] > 0) for run in runs) / len(runs),
            4,
        ),
        "recipient_informed_rate": round(
            sum(int(run["recipient_informed"]) for run in runs) / len(runs),
            4,
        ),
        "recipient_reached_food_rate": round(
            sum(int(run["recipient_reached_food"]) for run in runs) / len(runs),
            4,
        ),
        "recipient_foraged_rate": round(
            sum(int(run["recipient_forage_count"] > 0) for run in runs) / len(runs),
            4,
        ),
        "mean_recipient_final_distance": round(
            sum(int(run["recipient_final_distance"]) for run in runs) / len(runs),
            3,
        ),
        "mean_recipient_final_energy": round(
            sum(int(run["recipient_final_energy"]) for run in runs) / len(runs),
            3,
        ),
        "mean_sender_final_energy": round(
            sum(int(run["sender_final_energy"]) for run in runs) / len(runs),
            3,
        ),
        "total_signal_energy_spent": sum(int(run["signal_energy_spent"]) for run in runs),
    }


def _paired_effects(
    runs: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    by_pair = {(int(run["seed"]), str(run["condition"])): run for run in runs}
    normal_food_advantage = []
    normal_distance_advantage = []
    normal_energy_advantage = []
    for seed in seeds:
        normal = by_pair[(seed, "normal")]
        blocked = by_pair[(seed, "blocked")]
        normal_food_advantage.append(
            int(normal["recipient_forage_count"] > 0)
            - int(blocked["recipient_forage_count"] > 0)
        )
        normal_distance_advantage.append(
            int(blocked["recipient_final_distance"])
            - int(normal["recipient_final_distance"])
        )
        normal_energy_advantage.append(
            int(normal["recipient_final_energy"])
            - int(blocked["recipient_final_energy"])
        )
    return {
        "normal_minus_blocked_forage_rate": round(
            sum(normal_food_advantage) / len(seeds),
            4,
        ),
        "normal_distance_advantage": round(
            sum(normal_distance_advantage) / len(seeds),
            3,
        ),
        "normal_recipient_energy_advantage": round(
            sum(normal_energy_advantage) / len(seeds),
            3,
        ),
    }


def _scene_positions(seed: int) -> tuple[Position, Position, Position]:
    variants = (
        ((2, 2), (0, 3), (4, 3)),
        ((4, 2), (6, 3), (2, 3)),
        ((2, 2), (3, 0), (3, 4)),
        ((2, 4), (3, 6), (3, 2)),
    )
    return variants[seed % len(variants)]


def _v4_scene_positions(
    seed: int,
    *,
    width: int,
    height: int,
    radius: int,
) -> tuple[Position, Position, Position]:
    rng = Random(seed + 17_171)
    for _ in range(10_000):
        sender = (rng.randrange(width), rng.randrange(height))
        nearby = [
            (x, y)
            for x in range(max(0, sender[0] - radius), min(width, sender[0] + radius + 1))
            for y in range(max(0, sender[1] - radius), min(height, sender[1] + radius + 1))
            if (x, y) != sender and _manhattan(sender, (x, y)) <= radius
        ]
        targets = [
            position for position in nearby if _manhattan(sender, position) >= 2
        ]
        if not targets:
            continue
        target = rng.choice(targets)
        recipients = [
            position
            for position in nearby
            if position != target
            and _manhattan(position, target) > radius
            and _greedy_path_reaches(
                position,
                target,
                blocked=sender,
                width=width,
                height=height,
                steps=radius * 2 + 2,
            )
        ]
        if recipients:
            return sender, rng.choice(recipients), target
    raise RuntimeError("could not construct a diverse V4 communication scene")


def _v41_scene_positions(
    seed: int,
    *,
    width: int,
    height: int,
    radius: int,
) -> tuple[Position, Position, Position, Position, tuple[Position, Position]]:
    rng = Random(seed + 29_471)
    cells = [(x, y) for x in range(width) for y in range(height)]
    for _ in range(20_000):
        sender = rng.choice(cells)
        nearby = [
            cell
            for cell in cells
            if cell != sender and _manhattan(sender, cell) <= radius
        ]
        recipient = rng.choice(nearby)
        targets = [
            cell
            for cell in nearby
            if cell != recipient
            and _manhattan(sender, cell) >= 2
            and _manhattan(recipient, cell) > radius
        ]
        if not targets:
            continue
        target = rng.choice(targets)
        extras = [
            cell
            for cell in nearby
            if cell not in {recipient, target}
            and _manhattan(recipient, cell) > radius
            and _manhattan(sender, cell) > _manhattan(sender, target)
        ]
        if not extras:
            continue
        extra = rng.choice(extras)
        direct_path = _direct_greedy_path(recipient, target, width, height)
        distractor_candidates = [
            cell
            for cell in nearby
            if cell not in {recipient, target, extra}
            and cell not in direct_path
            and (
                _manhattan(cell, target) <= radius
                or _manhattan(cell, extra) <= radius
            )
        ]
        if len(distractor_candidates) < 2:
            continue
        rng.shuffle(distractor_candidates)
        distractors = (
            distractor_candidates[0],
            distractor_candidates[1],
        )
        if _greedy_path_reaches_many(
            recipient,
            target,
            blocked={sender, *distractors},
            width=width,
            height=height,
            steps=radius * 2 + 3,
        ):
            return sender, recipient, target, extra, distractors
    raise RuntimeError("could not construct a rich V4.1 communication scene")


def _direct_greedy_path(
    start: Position,
    target: Position,
    width: int,
    height: int,
) -> set[Position]:
    position = start
    path: set[Position] = set()
    for _ in range(width + height):
        if position == target:
            break
        candidates = [
            candidate
            for candidate in (
                (position[0] - 1, position[1]),
                (position[0] + 1, position[1]),
                (position[0], position[1] - 1),
                (position[0], position[1] + 1),
            )
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height
        ]
        position = min(
            candidates,
            key=lambda candidate: (_manhattan(candidate, target), candidate),
        )
        path.add(position)
    return path


def _greedy_path_reaches_many(
    start: Position,
    target: Position,
    *,
    blocked: set[Position],
    width: int,
    height: int,
    steps: int,
) -> bool:
    position = start
    for _ in range(steps):
        if position == target:
            return True
        candidates = [
            candidate
            for candidate in (
                (position[0] - 1, position[1]),
                (position[0] + 1, position[1]),
                (position[0], position[1] - 1),
                (position[0], position[1] + 1),
            )
            if 0 <= candidate[0] < width
            and 0 <= candidate[1] < height
            and candidate not in blocked
        ]
        if not candidates:
            return False
        position = min(
            candidates,
            key=lambda candidate: (_manhattan(candidate, target), candidate),
        )
    return position == target


def _greedy_path_reaches(
    start: Position,
    target: Position,
    *,
    blocked: Position,
    width: int,
    height: int,
    steps: int,
) -> bool:
    position = start
    for _ in range(steps):
        if position == target:
            return True
        neighbors = [
            candidate
            for candidate in (
                (position[0] - 1, position[1]),
                (position[0] + 1, position[1]),
                (position[0], position[1] - 1),
                (position[0], position[1] + 1),
            )
            if 0 <= candidate[0] < width
            and 0 <= candidate[1] < height
            and candidate != blocked
        ]
        best_distance = min(_manhattan(candidate, target) for candidate in neighbors)
        position = min(
            candidate
            for candidate in neighbors
            if _manhattan(candidate, target) == best_distance
        )
    return position == target


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


class _PairedCommunicationPolicy:
    def __init__(
        self,
        delegate: Policy,
        sender_id: str,
        sender_action_tick: int,
        forced_rest_ids: tuple[str, ...] = (),
    ) -> None:
        self.delegate = delegate
        self.sender_id = sender_id
        self.sender_action_tick = sender_action_tick
        self.forced_rest_ids = set(forced_rest_ids)
        self.last_raw_output: str | None = None

    def choose(self, observation: Observation) -> Action:
        if observation.organism_id in self.forced_rest_ids or (
            observation.organism_id == self.sender_id
            and observation.tick > self.sender_action_tick
        ):
            action = Action(ActionKind.REST)
            self.last_raw_output = json.dumps(action.to_dict(), separators=(",", ":"))
            return action
        action = self.delegate.choose(observation)
        self.last_raw_output = getattr(
            self.delegate,
            "last_raw_output",
            json.dumps(action.to_dict(), separators=(",", ":")),
        )
        return action

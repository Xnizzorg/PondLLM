from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable

from .domain import Action, ActionKind, Genes, Observation, Position
from .policies import Policy
from .prompting import strict_json_action
from .world import World, WorldConfig


RESCUE_CONDITIONS = ("normal", "blocked", "unsafe")


@dataclass(frozen=True, slots=True)
class RescueScene:
    seed: int
    initial_tick: int
    child_id: str
    donor_id: str
    distractor_id: str
    child_position: Position
    donor_position: Position
    distractor_position: Position
    target_food: Position
    child_energy: int
    safe_donor_energy: int
    unsafe_donor_energy: int
    share_threshold: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_rescue_world(seed: int, condition: str) -> tuple[World, RescueScene]:
    """Create a neutral low-energy rescue scene with a visible adjacent food cell."""

    if condition not in RESCUE_CONDITIONS:
        raise ValueError(f"unknown rescue condition: {condition}")
    variants = (
        ((4, 4), (3, 4), (4, 5), (1, 4)),
        ((4, 4), (4, 3), (5, 4), (4, 7)),
        ((4, 4), (5, 4), (4, 3), (7, 4)),
        ((4, 4), (4, 5), (3, 4), (4, 1)),
    )
    child_position, donor_position, target_food, distractor_position = variants[
        seed % len(variants)
    ]
    config = WorldConfig(
        width=9,
        height=9,
        founders=3,
        initial_energy=10,
        initial_food=0,
        food_energy=4,
        food_regrowth=0,
        max_food=8,
        metabolism=1,
        move_cost=0,
        rest_gain=1,
        reproduction_cost=8,
        child_energy=4,
        max_population=3,
        memory_limit=12,
        perception_radius=3,
        signal_delivery="normal",
        signal_cost=0,
    )
    world = World(config, seed=seed, initialize=False)
    initial_tick = 4 + (seed * 13) % 101
    world.tick = initial_tick
    rng = Random(seed + 84_211)
    organism_ids = [
        f"organism-{seed:06d}-{index:05d}" for index in range(1, 4)
    ]
    lineages = [
        f"lineage-{(seed * 43 + index) % 10_000_000:07d}"
        for index in range(1, 4)
    ]
    rng.shuffle(organism_ids)
    rng.shuffle(lineages)
    genes = Genes()
    safe_donor_energy = genes.share_threshold + 3 + seed % 3
    unsafe_donor_energy = genes.share_threshold - 1
    child = world.spawn_founder(
        child_position,
        lineages[0],
        energy=1,
        genes=Genes(),
        organism_id=organism_ids[0],
    )
    child.age = min(initial_tick, 1 + seed % 5)
    donor = world.spawn_founder(
        donor_position,
        lineages[1],
        energy=(
            unsafe_donor_energy
            if condition == "unsafe"
            else safe_donor_energy
        ),
        genes=genes,
        organism_id=organism_ids[1],
    )
    donor.age = min(initial_tick, 8 + seed % 17)
    distractor = world.spawn_founder(
        distractor_position,
        lineages[2],
        energy=9,
        genes=Genes(),
        organism_id=organism_ids[2],
    )
    distractor.age = min(initial_tick, 4 + seed % 11)
    world.add_food(target_food, 3)
    return world, RescueScene(
        seed=seed,
        initial_tick=initial_tick,
        child_id=child.organism_id,
        donor_id=donor.organism_id,
        distractor_id=distractor.organism_id,
        child_position=child_position,
        donor_position=donor_position,
        distractor_position=distractor_position,
        target_food=target_food,
        child_energy=1,
        safe_donor_energy=safe_donor_energy,
        unsafe_donor_energy=unsafe_donor_energy,
        share_threshold=genes.share_threshold,
    )


def run_rescue_experiment(
    policy: Policy | None,
    output_dir: str | Path,
    seeds: Iterable[int],
    steps: int = 5,
    metadata: dict[str, Any] | None = None,
    policy_factory: Callable[[int, str], Policy] | None = None,
) -> dict[str, Any]:
    """Run safe, share-blocked, and unsafe-donor rescue interventions."""

    if steps < 1:
        raise ValueError("steps must be positive")
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("rescue experiment seeds must be unique")
    if (policy is None) == (policy_factory is None):
        raise ValueError("provide exactly one of policy or policy_factory")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in seed_values:
        safe_scene: dict[str, Any] | None = None
        for condition in RESCUE_CONDITIONS:
            world, scene = create_rescue_world(seed, condition)
            scene_payload = scene.to_dict()
            if condition in {"normal", "blocked"}:
                if safe_scene is None:
                    safe_scene = scene_payload
                elif scene_payload != safe_scene:
                    raise RuntimeError("paired rescue worlds do not share an initial scene")
            delegate = (
                policy_factory(seed, condition)
                if policy_factory is not None
                else policy
            )
            if delegate is None:
                raise RuntimeError("rescue policy was not constructed")
            wrapper = _RescuePolicy(
                delegate=delegate,
                donor_id=scene.donor_id,
                child_id=scene.child_id,
                distractor_id=scene.distractor_id,
                block_shares=condition == "blocked",
            )
            world.run(wrapper, steps)
            run_summary = _summarize_rescue(world, scene, condition, wrapper)
            run_path = destination / f"seed-{seed:05d}-{condition}"
            run_path.mkdir(parents=True, exist_ok=True)
            world.write_events(run_path / "events.jsonl")
            with (run_path / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(run_summary, handle, indent=2)
            runs.append(run_summary)

    summary = {
        "seeds": seed_values,
        "steps": steps,
        "conditions": list(RESCUE_CONDITIONS),
        "metadata": metadata or {},
        "runs": runs,
        "per_condition": {
            condition: _aggregate_condition(
                [run for run in runs if run["condition"] == condition]
            )
            for condition in RESCUE_CONDITIONS
        },
        "paired_effects": _paired_effects(runs, seed_values),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _summarize_rescue(
    world: World,
    scene: RescueScene,
    condition: str,
    wrapper: "_RescuePolicy",
) -> dict[str, Any]:
    decisions = [
        event
        for event in world.events
        if event.kind == "decision"
        and event.actor_id in {scene.child_id, scene.donor_id}
    ]
    child_decisions = [
        event for event in decisions if event.actor_id == scene.child_id
    ]
    donor_decisions = [
        event for event in decisions if event.actor_id == scene.donor_id
    ]
    shares = [
        event
        for event in world.events
        if event.kind == "share"
        and event.actor_id == scene.donor_id
        and event.data.get("target") == scene.child_id
    ]
    child_forages = [
        event
        for event in world.events
        if event.kind == "forage" and event.actor_id == scene.child_id
    ]
    child_boundary_moves = [
        event
        for event in child_decisions
        if event.data["action"]["action"] == ActionKind.MOVE.value
        and int(event.data["observation"]["self"]["energy"])
        <= _manhattan(
            tuple(event.data["observation"]["self"]["position"]),
            scene.target_food,
        )
    ]
    child_waits = [
        event
        for event in child_decisions
        if event.data["action"]["action"] == ActionKind.REST.value
        and int(event.data["observation"]["self"]["energy"]) == 1
    ]
    donor_energies = [
        int(event.data["observation"]["self"]["energy"])
        for event in donor_decisions
    ]
    donor = world.organisms[scene.donor_id]
    child = world.organisms[scene.child_id]
    strict_outputs = sum(
        strict_json_action(event.data.get("raw_output")) is not None
        for event in decisions
    )
    return {
        "seed": scene.seed,
        "condition": condition,
        "scene": scene.to_dict(),
        "model_decisions": len(decisions),
        "valid_model_decisions": sum(
            bool(event.data["valid"]) and "policy_error" not in event.data
            for event in decisions
        ),
        "strict_model_outputs": strict_outputs,
        "model_policy_errors": sum(
            "policy_error" in event.data for event in decisions
        ),
        "child_actions": dict(
            sorted(
                Counter(
                    str(event.data["action"]["action"])
                    for event in child_decisions
                ).items()
            )
        ),
        "donor_actions": dict(
            sorted(
                Counter(
                    str(event.data["action"]["action"])
                    for event in donor_decisions
                ).items()
            )
        ),
        "donor_share_count": len(shares),
        "donor_repeated_shares": max(0, len(shares) - 1),
        "blocked_share_attempts": wrapper.blocked_share_attempts,
        "unsafe_share_count": len(shares) if condition == "unsafe" else 0,
        "child_energy_one_waits": len(child_waits),
        "child_boundary_moves": len(child_boundary_moves),
        "child_reached_food": child.position == scene.target_food
        or any(
            tuple(event.data["observation"]["self"]["position"])
            == scene.target_food
            for event in child_decisions
        ),
        "child_forage_count": len(child_forages),
        "child_alive": child.alive,
        "child_final_position": list(child.position),
        "child_final_energy": child.energy,
        "donor_min_energy": min([donor.energy, *donor_energies]),
        "donor_final_energy": donor.energy,
        "donor_alive": donor.alive,
        "world": world.summary(),
    }


def _aggregate_condition(runs: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = sum(int(run["model_decisions"]) for run in runs)
    return {
        "runs": len(runs),
        "model_decisions": decisions,
        "model_action_valid_rate": round(
            sum(int(run["valid_model_decisions"]) for run in runs) / decisions,
            4,
        ),
        "strict_json_rate": round(
            sum(int(run["strict_model_outputs"]) for run in runs) / decisions,
            4,
        ),
        "model_policy_error_rate": round(
            sum(int(run["model_policy_errors"]) for run in runs) / decisions,
            4,
        ),
        "donor_share_rate": round(
            sum(int(run["donor_share_count"] > 0) for run in runs) / len(runs),
            4,
        ),
        "child_wait_rate": round(
            sum(int(run["child_energy_one_waits"] > 0) for run in runs)
            / len(runs),
            4,
        ),
        "child_forage_rate": round(
            sum(int(run["child_forage_count"] > 0) for run in runs) / len(runs),
            4,
        ),
        "child_survival_rate": round(
            sum(int(run["child_alive"]) for run in runs) / len(runs),
            4,
        ),
        "total_repeated_shares": sum(
            int(run["donor_repeated_shares"]) for run in runs
        ),
        "total_unsafe_shares": sum(
            int(run["unsafe_share_count"]) for run in runs
        ),
        "total_child_boundary_moves": sum(
            int(run["child_boundary_moves"]) for run in runs
        ),
        "mean_donor_min_energy": round(
            sum(int(run["donor_min_energy"]) for run in runs) / len(runs),
            3,
        ),
        "mean_donor_final_energy": round(
            sum(int(run["donor_final_energy"]) for run in runs) / len(runs),
            3,
        ),
    }


def _paired_effects(
    runs: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    by_pair = {(int(run["seed"]), str(run["condition"])): run for run in runs}
    return {
        "normal_minus_blocked_forage_rate": round(
            sum(
                int(by_pair[(seed, "normal")]["child_forage_count"] > 0)
                - int(by_pair[(seed, "blocked")]["child_forage_count"] > 0)
                for seed in seeds
            )
            / len(seeds),
            4,
        ),
        "normal_minus_blocked_child_final_energy": round(
            sum(
                int(by_pair[(seed, "normal")]["child_final_energy"])
                - int(by_pair[(seed, "blocked")]["child_final_energy"])
                for seed in seeds
            )
            / len(seeds),
            3,
        ),
    }


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


class _RescuePolicy:
    def __init__(
        self,
        delegate: Policy,
        donor_id: str,
        child_id: str,
        distractor_id: str,
        *,
        block_shares: bool,
    ) -> None:
        self.delegate = delegate
        self.donor_id = donor_id
        self.child_id = child_id
        self.distractor_id = distractor_id
        self.block_shares = block_shares
        self.blocked_share_attempts = 0
        self.last_raw_output: str | None = None

    def choose(self, observation: Observation) -> Action:
        if observation.organism_id == self.distractor_id:
            action = Action(ActionKind.REST)
            self.last_raw_output = json.dumps(action.to_dict(), separators=(",", ":"))
            return action
        action = self.delegate.choose(observation)
        self.last_raw_output = getattr(
            self.delegate,
            "last_raw_output",
            json.dumps(action.to_dict(), separators=(",", ":")),
        )
        if (
            self.block_shares
            and observation.organism_id == self.donor_id
            and action.kind is ActionKind.SHARE
            and action.target_id == self.child_id
        ):
            self.blocked_share_attempts += 1
            return Action(ActionKind.REST)
        return action

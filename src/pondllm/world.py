from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any

from .domain import Action, ActionKind, Event, Genes, Observation, Organism, Position

if TYPE_CHECKING:
    from .policies import Policy


@dataclass(frozen=True, slots=True)
class WorldConfig:
    width: int = 24
    height: int = 24
    founders: int = 12
    initial_energy: int = 10
    initial_food: int = 80
    food_energy: int = 4
    food_regrowth: int = 5
    max_food: int = 120
    metabolism: int = 1
    move_cost: int = 0
    rest_gain: int = 1
    reproduction_cost: int = 8
    child_energy: int = 4
    max_population: int = 48
    memory_limit: int = 12
    perception_radius: int = 3
    signal_delivery: str = "normal"
    signal_cost: int = 0

    def validate(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("world must be at least 3x3")
        if self.founders < 1 or self.founders > self.width * self.height:
            raise ValueError("founder count does not fit world")
        if self.max_population < self.founders:
            raise ValueError("max_population must be at least founders")
        if self.reproduction_cost < self.child_energy:
            raise ValueError("reproduction_cost must cover child_energy")
        if self.signal_delivery not in {"normal", "blocked", "corrupted"}:
            raise ValueError("signal_delivery must be normal, blocked, or corrupted")
        if self.signal_cost < 0:
            raise ValueError("signal_cost must be non-negative")


class World:
    def __init__(self, config: WorldConfig, seed: int, initialize: bool = True) -> None:
        config.validate()
        self.config = config
        self.seed = seed
        self.rng = Random(seed)
        self.tick = 0
        self.organisms: dict[str, Organism] = {}
        self.food: dict[Position, int] = {}
        self.events: list[Event] = []
        self._next_organism_number = 1
        if initialize:
            self._initialize()

    def _initialize(self) -> None:
        cells = [(x, y) for x in range(self.config.width) for y in range(self.config.height)]
        self.rng.shuffle(cells)
        for index in range(self.config.founders):
            self.spawn_founder(
                position=cells[index],
                lineage_id=f"lineage-{index + 1:02d}",
            )
        self._grow_food(self.config.initial_food)

    def spawn_founder(
        self,
        position: Position,
        lineage_id: str,
        energy: int | None = None,
        genes: Genes | None = None,
        organism_id: str | None = None,
    ) -> Organism:
        if not self.in_bounds(position):
            raise ValueError("founder position is out of bounds")
        if self.organism_at(position) is not None:
            raise ValueError("founder position is occupied")
        if organism_id is not None and organism_id in self.organisms:
            raise ValueError("founder organism id is already in use")
        organism = Organism(
            organism_id=self._new_organism_id() if organism_id is None else organism_id,
            lineage_id=lineage_id,
            position=position,
            energy=self.config.initial_energy if energy is None else energy,
            genes=genes or Genes(),
        )
        self.organisms[organism.organism_id] = organism
        self._emit("founder", organism.organism_id, {"lineage": lineage_id, "position": position})
        return organism

    def add_food(self, position: Position, amount: int = 1) -> None:
        if not self.in_bounds(position):
            raise ValueError("food position is out of bounds")
        if amount < 1:
            raise ValueError("food amount must be positive")
        self.food[position] = self.food.get(position, 0) + amount

    def living(self) -> list[Organism]:
        return [organism for organism in self.organisms.values() if organism.alive]

    def organism_at(self, position: Position) -> Organism | None:
        for organism in self.organisms.values():
            if organism.alive and organism.position == position:
                return organism
        return None

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def neighbors(self, position: Position) -> list[Position]:
        x, y = position
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [candidate for candidate in candidates if self.in_bounds(candidate)]

    def observe(self, organism: Organism) -> Observation:
        radius = self.config.perception_radius
        visible_food = []
        for (x, y), amount in self.food.items():
            if amount > 0 and _manhattan(organism.position, (x, y)) <= radius:
                visible_food.append((x, y, amount))
        visible_food.sort(key=lambda item: (_manhattan(organism.position, (item[0], item[1])), item))

        visible_agents: list[dict[str, Any]] = []
        for other in self.living():
            distance = _manhattan(organism.position, other.position)
            if other.organism_id != organism.organism_id and distance <= radius:
                visible_agents.append(
                    {
                        "id": other.organism_id,
                        "lineage": other.lineage_id,
                        "position": list(other.position),
                        "energy": other.energy,
                        "distance": distance,
                    }
                )
        visible_agents.sort(key=lambda item: (item["distance"], item["id"]))

        open_neighbors = tuple(
            position for position in self.neighbors(organism.position) if self.organism_at(position) is None
        )
        return Observation(
            tick=self.tick,
            perception_radius=radius,
            organism_id=organism.organism_id,
            lineage_id=organism.lineage_id,
            position=organism.position,
            energy=organism.energy,
            age=organism.age,
            drives=organism.genes.public_drives(),
            current_food=self.food.get(organism.position, 0),
            visible_food=tuple(visible_food),
            visible_agents=tuple(visible_agents),
            open_neighbors=open_neighbors,
            memory=tuple(organism.memory),
        )

    def step(self, policy: "Policy") -> None:
        actor_ids = [organism.organism_id for organism in self.living()]
        self.rng.shuffle(actor_ids)
        for organism_id in actor_ids:
            organism = self.organisms[organism_id]
            if not organism.alive:
                continue
            observation = self.observe(organism)
            policy_error: str | None = None
            try:
                action = policy.choose(observation)
            except Exception as exc:  # A policy failure must not terminate an ecological run.
                policy_error = f"{type(exc).__name__}: {exc}"
                action = Action(ActionKind.REST)

            valid, result = self.apply_action(organism, action)
            decision_data: dict[str, Any] = {
                "observation": observation.to_dict(),
                "action": action.to_dict(),
                "valid": valid,
                "result": result,
            }
            if policy_error:
                decision_data["policy_error"] = policy_error
            raw_output = getattr(policy, "last_raw_output", None)
            if raw_output is not None:
                decision_data["raw_output"] = raw_output
            self._emit("decision", organism.organism_id, decision_data)

            organism.age += 1
            organism.energy -= self.config.metabolism
            if organism.energy <= 0:
                organism.alive = False
                self._emit(
                    "death",
                    organism.organism_id,
                    {"age": organism.age, "lineage": organism.lineage_id},
                )

        self._grow_food(self.config.food_regrowth)
        self.tick += 1

    def run(self, policy: "Policy", steps: int) -> dict[str, Any]:
        for _ in range(steps):
            if not self.living():
                break
            self.step(policy)
        return self.summary()

    def apply_action(self, organism: Organism, action: Action) -> tuple[bool, str]:
        if not organism.alive:
            return False, "actor is dead"

        if action.kind is ActionKind.FORAGE:
            amount = self.food.get(organism.position, 0)
            if amount < 1:
                return False, "no food at current position"
            organism.energy += self.config.food_energy
            if amount == 1:
                del self.food[organism.position]
            else:
                self.food[organism.position] = amount - 1
            self._emit("forage", organism.organism_id, {"position": organism.position})
            return True, f"gained {self.config.food_energy} energy"

        if action.kind is ActionKind.MOVE:
            target = action.target_position
            if target not in self.neighbors(organism.position):
                return False, "move target is not an adjacent cell"
            if self.organism_at(target) is not None:
                return False, "move target is occupied"
            origin = organism.position
            organism.position = target
            organism.energy -= self.config.move_cost
            self._emit("move", organism.organism_id, {"from": origin, "to": target})
            return True, "moved"

        if action.kind is ActionKind.SHARE:
            target = self.organisms.get(action.target_id or "")
            amount = action.amount or 0
            if target is None or not target.alive:
                return False, "share target is not alive"
            if _manhattan(organism.position, target.position) != 1:
                return False, "share target is not adjacent"
            if organism.energy <= amount:
                return False, "insufficient energy"
            organism.energy -= amount
            target.energy += amount
            target.remember(f"{organism.organism_id} shared {amount} energy", self.config.memory_limit)
            self._emit("share", organism.organism_id, {"target": target.organism_id, "amount": amount})
            return True, "shared energy"

        if action.kind is ActionKind.SIGNAL:
            if organism.energy <= self.config.signal_cost:
                return False, "insufficient energy for signal cost"
            organism.energy -= self.config.signal_cost
            delivered_message = action.message or ""
            if self.config.signal_delivery == "corrupted":
                delivered_message = _corrupt_food_coordinate(
                    delivered_message,
                    width=self.config.width,
                    height=self.config.height,
                )
            receivers = []
            if self.config.signal_delivery != "blocked":
                for receiver in self.living():
                    if receiver.organism_id == organism.organism_id:
                        continue
                    if (
                        _manhattan(organism.position, receiver.position)
                        <= self.config.perception_radius
                    ):
                        receiver.remember(
                            f"{organism.organism_id} signalled: {delivered_message}",
                            self.config.memory_limit,
                        )
                        receivers.append(receiver.organism_id)
            self._emit(
                "signal",
                organism.organism_id,
                {
                    "message": action.message,
                    "delivered_message": (
                        delivered_message if self.config.signal_delivery != "blocked" else None
                    ),
                    "receivers": receivers,
                    "delivery": self.config.signal_delivery,
                    "cost": self.config.signal_cost,
                },
            )
            return True, f"signal reached {len(receivers)} organisms"

        if action.kind is ActionKind.REPRODUCE:
            target = action.target_position
            if len(self.living()) >= self.config.max_population:
                return False, "population cap reached"
            if organism.energy < max(organism.genes.reproduction_threshold, self.config.reproduction_cost + 1):
                return False, "energy below reproduction threshold"
            if target not in self.neighbors(organism.position):
                return False, "offspring target is not adjacent"
            if self.organism_at(target) is not None:
                return False, "offspring target is occupied"
            organism.energy -= self.config.reproduction_cost
            child = Organism(
                organism_id=self._new_organism_id(),
                lineage_id=organism.lineage_id,
                position=target,
                energy=self.config.child_energy,
                genes=organism.genes.mutate(self.rng),
                parent_id=organism.organism_id,
                generation=organism.generation + 1,
            )
            self.organisms[child.organism_id] = child
            self._emit(
                "birth",
                organism.organism_id,
                {
                    "child_id": child.organism_id,
                    "lineage": child.lineage_id,
                    "position": target,
                    "generation": child.generation,
                    "genes": asdict(child.genes),
                },
            )
            return True, f"created {child.organism_id}"

        if action.kind is ActionKind.REST:
            organism.energy += self.config.rest_gain
            self._emit("rest", organism.organism_id)
            return True, f"recovered {self.config.rest_gain} energy"

        return False, "unsupported action"

    def summary(self) -> dict[str, Any]:
        living = self.living()
        lineages = Counter(organism.lineage_id for organism in living)
        return {
            "seed": self.seed,
            "ticks": self.tick,
            "born": sum(event.kind == "birth" for event in self.events),
            "died": sum(event.kind == "death" for event in self.events),
            "living": len(living),
            "total_organisms": len(self.organisms),
            "living_lineages": dict(sorted(lineages.items())),
            "mean_living_energy": (
                round(sum(organism.energy for organism in living) / len(living), 3) if living else 0.0
            ),
            "food_units": sum(self.food.values()),
        }

    def write_events(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def _new_organism_id(self) -> str:
        while True:
            organism_id = f"organism-{self._next_organism_number:05d}"
            self._next_organism_number += 1
            if organism_id not in self.organisms:
                return organism_id

    def _grow_food(self, requested_units: int) -> None:
        capacity = self.config.max_food - sum(self.food.values())
        for _ in range(max(0, min(requested_units, capacity))):
            position = (
                self.rng.randrange(self.config.width),
                self.rng.randrange(self.config.height),
            )
            self.food[position] = self.food.get(position, 0) + 1

    def _emit(self, kind: str, actor_id: str | None, data: dict[str, Any] | None = None) -> None:
        self.events.append(Event(tick=self.tick, kind=kind, actor_id=actor_id, data=data or {}))


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


_FOOD_COORDINATE = re.compile(r"food at \[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", re.IGNORECASE)


def _corrupt_food_coordinate(message: str, width: int, height: int) -> str:
    match = _FOOD_COORDINATE.search(message)
    if match is None:
        return f"corrupted: {message}"
    x = (int(match.group(1)) + max(1, width // 2)) % width
    y = (int(match.group(2)) + max(1, height // 2)) % height
    return _FOOD_COORDINATE.sub(f"food at [{x},{y}]", message, count=1)

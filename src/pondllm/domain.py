from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from random import Random
from typing import Any


Position = tuple[int, int]


class ActionKind(str, Enum):
    MOVE = "move"
    FORAGE = "forage"
    SHARE = "share"
    SIGNAL = "signal"
    REST = "rest"
    REPRODUCE = "reproduce"


class InvalidAction(ValueError):
    """Raised when a policy emits an action that does not match the public protocol."""


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    target_position: Position | None = None
    target_id: str | None = None
    amount: int | None = None
    message: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Action":
        if not isinstance(payload, dict):
            raise InvalidAction("action payload must be an object")
        try:
            kind = ActionKind(payload.get("action"))
        except (TypeError, ValueError) as exc:
            raise InvalidAction("unknown or missing action") from exc

        if kind in {ActionKind.MOVE, ActionKind.REPRODUCE}:
            raw_target = payload.get("target")
            if (
                not isinstance(raw_target, (list, tuple))
                or len(raw_target) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_target)
            ):
                raise InvalidAction(f"{kind.value} requires target [x, y]")
            return cls(kind=kind, target_position=(raw_target[0], raw_target[1]))

        if kind is ActionKind.SHARE:
            target_id = payload.get("target")
            amount = payload.get("amount")
            if not isinstance(target_id, str) or not target_id:
                raise InvalidAction("share requires a target organism id")
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 1:
                raise InvalidAction("share amount must be a positive integer")
            return cls(kind=kind, target_id=target_id, amount=amount)

        if kind is ActionKind.SIGNAL:
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                raise InvalidAction("signal requires a non-empty message")
            return cls(kind=kind, message=message.strip()[:160])

        return cls(kind=kind)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.kind.value}
        if self.target_position is not None:
            payload["target"] = list(self.target_position)
        if self.target_id is not None:
            payload["target"] = self.target_id
        if self.amount is not None:
            payload["amount"] = self.amount
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass(slots=True)
class Genes:
    reproduction_threshold: int = 16
    share_threshold: int = 12
    curiosity: float = 0.35
    sociality: float = 0.25
    fecundity: float = 0.20

    def mutate(self, rng: Random) -> "Genes":
        return Genes(
            reproduction_threshold=_bounded_int(
                self.reproduction_threshold + rng.choice((-1, 0, 0, 0, 1)), 10, 30
            ),
            share_threshold=_bounded_int(
                self.share_threshold + rng.choice((-1, 0, 0, 0, 1)), 7, 25
            ),
            curiosity=_bounded_float(self.curiosity + rng.gauss(0.0, 0.025)),
            sociality=_bounded_float(self.sociality + rng.gauss(0.0, 0.025)),
            fecundity=_bounded_float(self.fecundity + rng.gauss(0.0, 0.025)),
        )

    def public_drives(self) -> dict[str, float | int]:
        return {
            "reproduction_threshold": self.reproduction_threshold,
            "share_threshold": self.share_threshold,
            "curiosity": round(self.curiosity, 3),
            "sociality": round(self.sociality, 3),
            "fecundity": round(self.fecundity, 3),
        }


@dataclass(slots=True)
class Organism:
    organism_id: str
    lineage_id: str
    position: Position
    energy: int
    genes: Genes = field(default_factory=Genes)
    parent_id: str | None = None
    generation: int = 0
    age: int = 0
    alive: bool = True
    memory: list[str] = field(default_factory=list)

    def remember(self, item: str, limit: int) -> None:
        self.memory.append(item[:240])
        if len(self.memory) > limit:
            del self.memory[: len(self.memory) - limit]


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    perception_radius: int
    organism_id: str
    lineage_id: str
    position: Position
    energy: int
    age: int
    drives: dict[str, float | int]
    current_food: int
    visible_food: tuple[tuple[int, int, int], ...]
    visible_agents: tuple[dict[str, Any], ...]
    open_neighbors: tuple[Position, ...]
    memory: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "perception_radius": self.perception_radius,
            "self": {
                "id": self.organism_id,
                "lineage": self.lineage_id,
                "position": list(self.position),
                "energy": self.energy,
                "age": self.age,
                "drives": self.drives,
            },
            "current_food": self.current_food,
            "visible_food": [
                {"position": [x, y], "amount": amount} for x, y, amount in self.visible_food
            ],
            "visible_agents": list(self.visible_agents),
            "open_neighbors": [list(position) for position in self.open_neighbors],
            "memory": list(self.memory),
        }


@dataclass(frozen=True, slots=True)
class Event:
    tick: int
    kind: str
    actor_id: str | None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _bounded_float(value: float) -> float:
    return max(0.0, min(1.0, value))

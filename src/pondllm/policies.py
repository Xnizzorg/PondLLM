from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from .domain import Action, ActionKind, Observation, Position


class Policy(Protocol):
    def choose(self, observation: Observation) -> Action: ...


@dataclass(slots=True)
class HeuristicPolicy:
    seed: int = 0
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def choose(self, observation: Observation) -> Action:
        if observation.current_food > 0:
            return Action(ActionKind.FORAGE)

        drives = observation.drives
        open_neighbors = list(observation.open_neighbors)
        reproduction_threshold = int(drives["reproduction_threshold"])
        fecundity = float(drives["fecundity"])
        if (
            observation.energy >= reproduction_threshold
            and open_neighbors
            and self._rng.random() < fecundity
        ):
            return Action(ActionKind.REPRODUCE, target_position=self._rng.choice(open_neighbors))

        adjacent_agents = [agent for agent in observation.visible_agents if agent["distance"] == 1]
        sociality = float(drives["sociality"])
        share_threshold = int(drives["share_threshold"])
        needy = [agent for agent in adjacent_agents if agent["energy"] <= 3]
        if (
            observation.energy >= share_threshold
            and needy
            and self._rng.random() < sociality
        ):
            target = min(needy, key=lambda agent: (agent["energy"], agent["id"]))
            return Action(ActionKind.SHARE, target_id=target["id"], amount=1)

        if observation.visible_food and open_neighbors:
            food_position = observation.visible_food[0][:2]
            best_distance = min(_manhattan(position, food_position) for position in open_neighbors)
            best_moves = [
                position
                for position in open_neighbors
                if _manhattan(position, food_position) == best_distance
            ]
            return Action(ActionKind.MOVE, target_position=self._rng.choice(best_moves))

        if observation.visible_agents and self._rng.random() < sociality * 0.08:
            return Action(ActionKind.SIGNAL, message="food? energy? here")

        if open_neighbors and self._rng.random() < float(drives["curiosity"]):
            return Action(ActionKind.MOVE, target_position=self._rng.choice(open_neighbors))

        return Action(ActionKind.REST)


@dataclass(slots=True)
class RandomPolicy:
    seed: int = 0
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def choose(self, observation: Observation) -> Action:
        candidates = [Action(ActionKind.REST), Action(ActionKind.SIGNAL, message="here")]
        if observation.current_food > 0:
            candidates.append(Action(ActionKind.FORAGE))
        candidates.extend(
            Action(ActionKind.MOVE, target_position=position)
            for position in observation.open_neighbors
        )
        adjacent = [agent for agent in observation.visible_agents if agent["distance"] == 1]
        if adjacent and observation.energy > 2:
            candidates.append(
                Action(ActionKind.SHARE, target_id=self._rng.choice(adjacent)["id"], amount=1)
            )
        if (
            observation.energy >= int(observation.drives["reproduction_threshold"])
            and observation.open_neighbors
        ):
            candidates.append(
                Action(
                    ActionKind.REPRODUCE,
                    target_position=self._rng.choice(observation.open_neighbors),
                )
            )
        return self._rng.choice(candidates)


@dataclass(slots=True)
class DemonstrationPolicy:
    """A legal policy that adds action coverage to heuristic SFT demonstrations."""

    seed: int = 0
    _rng: Random = field(init=False, repr=False)
    _heuristic: HeuristicPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)
        self._heuristic = HeuristicPolicy(self.seed + 1)

    def choose(self, observation: Observation) -> Action:
        if observation.current_food > 0:
            return Action(ActionKind.FORAGE)

        adjacent = [agent for agent in observation.visible_agents if agent["distance"] == 1]
        if adjacent and observation.energy > 4 and self._rng.random() < 0.18:
            target = min(adjacent, key=lambda agent: (agent["energy"], agent["id"]))
            return Action(ActionKind.SHARE, target_id=target["id"], amount=1)

        if observation.visible_agents and self._rng.random() < 0.08:
            nearest = observation.visible_agents[0]
            message = f"at {list(observation.position)}; energy {observation.energy}"
            if nearest["energy"] <= 3:
                message = f"{nearest['id']} low energy"
            return Action(ActionKind.SIGNAL, message=message)

        return self._heuristic.choose(observation)


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])

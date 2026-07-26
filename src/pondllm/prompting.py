from __future__ import annotations

import json
from typing import Any

from .domain import Action, ActionKind, InvalidAction, Observation


SYSTEM_PROMPT = """You are one bounded organism in a resource ecology.
You know only the supplied local observation and your own memory. Never invent hidden world state.
Choose exactly one legal action and return only one JSON object.

Legal forms:
{"action":"move","target":[x,y]}
{"action":"forage"}
{"action":"share","target":"organism_id","amount":1}
{"action":"signal","message":"short message"}
{"action":"rest"}
{"action":"reproduce","target":[x,y]}

Move and reproduce targets must be listed in open_neighbors. Share targets must be visible agents.
Foraging only succeeds when current_food is positive. Energy is scarce and reaching zero is death.
Forage before signalling when food is on your current cell. Before moving toward food, compare
Manhattan distance with energy: you need energy for every move and one later forage action. Rest
instead when the food cannot be reached before exhaustion, unless an adjacent organism safely
helps. Share one energy only with an adjacent organism at energy 3 or lower when your own energy
is at or above share_threshold.
Signals are delivered to nearby organisms as memory. Report useful food locations as
{"action":"signal","message":"food at [x,y]"}. A remembered food coordinate may guide movement
when no food is currently visible. After receiving a useful coordinate, pursue or forage it
instead of relaying the same information. Use perception_radius and positions to avoid signalling
food a visible organism can already see."""


def observation_message(observation: Observation | dict[str, Any]) -> str:
    payload = observation.to_dict() if isinstance(observation, Observation) else observation
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def training_record(observation: Observation | dict[str, Any], action: Action) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_message(observation)},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(action.to_dict(), separators=(",", ":"), sort_keys=True),
            }
        ],
    }


def parse_action_text(text: str) -> Action:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        return Action.from_payload(payload)
    raise InvalidAction("model output did not contain a valid action object")


def strict_json_action(text: str | None) -> Action | None:
    """Return an action only when the complete raw output is one JSON object."""

    if not isinstance(text, str) or not text.strip():
        return None
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return None
        return Action.from_payload(payload)
    except (json.JSONDecodeError, InvalidAction, TypeError, ValueError):
        return None


def action_is_observation_legal(
    action: Action, observation: Observation | dict[str, Any]
) -> tuple[bool, str]:
    payload = observation.to_dict() if isinstance(observation, Observation) else observation
    self_state = payload["self"]
    open_neighbors = {tuple(position) for position in payload.get("open_neighbors", [])}

    if action.kind is ActionKind.FORAGE:
        return (True, "legal") if payload.get("current_food", 0) > 0 else (False, "no food")
    if action.kind is ActionKind.MOVE:
        return (
            (True, "legal")
            if action.target_position in open_neighbors
            else (False, "target not open")
        )
    if action.kind is ActionKind.REPRODUCE:
        if action.target_position not in open_neighbors:
            return False, "target not open"
        threshold = self_state.get("drives", {}).get("reproduction_threshold", 10**9)
        return (
            (True, "legal")
            if self_state.get("energy", 0) >= threshold
            else (False, "below reproduction threshold")
        )
    if action.kind is ActionKind.SHARE:
        candidates = {
            item["id"]: item for item in payload.get("visible_agents", []) if item.get("distance") == 1
        }
        if action.target_id not in candidates:
            return False, "target not adjacent"
        if self_state.get("energy", 0) <= (action.amount or 0):
            return False, "insufficient energy"
        return True, "legal"
    return True, "legal"

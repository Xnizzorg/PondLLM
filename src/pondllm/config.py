from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from .world import WorldConfig


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int = 7
    steps: int = 100


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    episodes: int = 64
    steps_per_episode: int = 60
    seed: int = 1000


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model: str = "Qwen/Qwen3-0.6B"
    max_length: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 2.0
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8


@dataclass(frozen=True, slots=True)
class AppConfig:
    world: WorldConfig
    run: RunConfig
    dataset: DatasetConfig
    training: TrainingConfig


T = TypeVar("T")


def load_config(path: str | Path = "configs/night_one.toml") -> AppConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return AppConfig(
        world=_from_mapping(WorldConfig, raw.get("world", {})),
        run=_from_mapping(RunConfig, raw.get("run", {})),
        dataset=_from_mapping(DatasetConfig, raw.get("dataset", {})),
        training=_from_mapping(TrainingConfig, raw.get("training", {})),
    )


def _from_mapping(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unexpected = set(values) - allowed
    if unexpected:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unexpected)}")
    return cls(**values)


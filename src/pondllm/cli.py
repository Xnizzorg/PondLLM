from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import load_config
from .curriculum_v4 import generate_v4_sft_dataset
from .curriculum_v41 import audit_v41_datasets, generate_v41_sft_dataset
from .dataset import (
    generate_balanced_sft_dataset,
    generate_communication_sft_dataset,
    generate_sft_dataset,
    stratify_sft_dataset,
)
from .policies import HeuristicPolicy, RandomPolicy
from .world import World


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pondllm", description="Local LLM ecology experiments")
    parser.add_argument("--config", default="configs/night_one.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="verify the installed GPU training stack")

    smoke = subparsers.add_parser("smoke", help="run a dependency-free ecology smoke test")
    smoke.add_argument("--steps", type=int)
    smoke.add_argument("--seed", type=int)
    smoke.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    smoke.add_argument("--founders", type=int)
    smoke.add_argument("--output", default="runs/smoke")

    dataset = subparsers.add_parser("dataset", help="generate heuristic SFT demonstrations")
    dataset.add_argument("--episodes", type=int)
    dataset.add_argument("--steps", type=int)
    dataset.add_argument("--seed", type=int)
    dataset.add_argument("--output", default="data/generated/sft.jsonl")

    balanced = subparsers.add_parser(
        "dataset-balanced",
        help="top up an SFT dataset with unique rare-action demonstrations",
    )
    balanced.add_argument("--base", default="data/generated/sft.jsonl")
    balanced.add_argument("--episodes", type=int, default=128)
    balanced.add_argument("--steps", type=int)
    balanced.add_argument("--seed", type=int, default=2000)
    balanced.add_argument("--minimum-per-action", type=int, default=1000)
    balanced.add_argument("--output", default="data/generated/sft-v2-balanced.jsonl")

    stratify = subparsers.add_parser(
        "dataset-stratify",
        help="sample equal held-out coverage for every action",
    )
    stratify.add_argument("--dataset", required=True)
    stratify.add_argument("--per-action", type=int, default=100)
    stratify.add_argument("--seed", type=int, default=23)
    stratify.add_argument("--output", default="data/generated/eval-v2-stratified.jsonl")

    communication = subparsers.add_parser(
        "dataset-communication",
        help="generate deterministic sender/recipient communication counterfactuals",
    )
    communication.add_argument("--base")
    communication.add_argument("--scenes", type=int, default=1000)
    communication.add_argument("--seed", type=int, default=3000)
    communication.add_argument("--output", default="data/generated/sft-v3-communication.jsonl")

    v4 = subparsers.add_parser(
        "dataset-v4",
        help="generate neutral simulator-native communication and survival SFT records",
    )
    v4.add_argument("--base")
    v4.add_argument("--scenes", type=int, default=1000)
    v4.add_argument("--survival-scenes", type=int, default=1000)
    v4.add_argument("--seed", type=int, default=52000)
    v4.add_argument("--no-trajectories", action="store_true")
    v4.add_argument("--output", default="data/generated/sft-v4-simulator-native.jsonl")

    v41 = subparsers.add_parser(
        "dataset-v41",
        help="extend V4 with rich communication and reachability SFT records",
    )
    v41.add_argument("--base")
    v41.add_argument("--rich-scenes", type=int, default=1000)
    v41.add_argument("--reachability-scenes", type=int, default=1000)
    v41.add_argument("--seed", type=int, default=61000)
    v41.add_argument("--no-trajectories", action="store_true")
    v41.add_argument(
        "--output",
        default="data/generated/sft-v4.1-rich-reachability.jsonl",
    )

    audit_v41 = subparsers.add_parser(
        "audit-v41",
        help="audit V4.1 legality, identity neutrality, and held-out overlap",
    )
    audit_v41.add_argument("--training", required=True)
    audit_v41.add_argument("--held-out", nargs="+", required=True)
    audit_v41.add_argument("--output")

    train = subparsers.add_parser("train", help="QLoRA fine-tune the action-policy adapter")
    train.add_argument("--dataset", default="data/generated/sft.jsonl")
    train.add_argument("--output", default="artifacts/qwen3-0.6b-action-sft")
    train.add_argument("--model")
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=float)
    train.add_argument("--max-length", type=int)
    train.add_argument("--micro-batch-size", type=int)
    train.add_argument("--gradient-accumulation-steps", type=int)

    run_model = subparsers.add_parser("run-model", help="run a local Qwen-controlled ecology")
    run_model.add_argument("--adapter")
    run_model.add_argument("--model")
    run_model.add_argument("--steps", type=int, default=30)
    run_model.add_argument("--seed", type=int)
    run_model.add_argument("--founders", type=int, default=4)
    run_model.add_argument("--temperature", type=float, default=0.7)
    run_model.add_argument("--load-in-4bit", action="store_true")
    run_model.add_argument("--output", default="runs/model-smoke")

    communication_world = subparsers.add_parser(
        "run-communication-world",
        help="run paired normal, blocked, corrupted, and costly communication worlds",
    )
    communication_world.add_argument("--adapter")
    communication_world.add_argument("--model")
    communication_world.add_argument("--steps", type=int, default=7)
    communication_world.add_argument("--scenes", type=int, default=4)
    communication_world.add_argument("--seed-start", type=int, default=100)
    communication_world.add_argument(
        "--profile",
        choices=("matched", "clean", "v4", "v41"),
        default="matched",
    )
    communication_world.add_argument(
        "--policy",
        choices=("model", "heuristic"),
        default="model",
    )
    communication_world.add_argument("--temperature", type=float, default=0.0)
    communication_world.add_argument("--load-in-4bit", action="store_true")
    communication_world.add_argument("--output", default="runs/communication-world")

    evaluate = subparsers.add_parser("evaluate", help="measure model action validity on held-out states")
    evaluate.add_argument("--dataset", default="data/generated/eval.jsonl")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--model")
    evaluate.add_argument("--max-records", type=int, default=200)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--load-in-4bit", action="store_true")
    evaluate.add_argument("--output", default="runs/evaluation/predictions.jsonl")

    evaluate_communication = subparsers.add_parser(
        "evaluate-communication",
        help="measure useful signalling and memory-conditioned recipient behavior",
    )
    evaluate_communication.add_argument("--dataset", required=True)
    evaluate_communication.add_argument("--adapter")
    evaluate_communication.add_argument("--model")
    evaluate_communication.add_argument("--temperature", type=float, default=0.0)
    evaluate_communication.add_argument("--load-in-4bit", action="store_true")
    evaluate_communication.add_argument(
        "--output",
        default="runs/communication-evaluation/predictions.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()

    config = load_config(args.config)
    if args.command == "smoke":
        steps = config.run.steps if args.steps is None else args.steps
        seed = config.run.seed if args.seed is None else args.seed
        policy = HeuristicPolicy(seed + 100) if args.policy == "heuristic" else RandomPolicy(seed + 100)
        world_config = config.world
        if args.founders is not None:
            world_config = replace(
                world_config,
                founders=args.founders,
                max_population=max(args.founders, min(world_config.max_population, args.founders * 3)),
            )
        world = World(world_config, seed=seed)
        summary = world.run(policy, steps)
        summary["policy"] = args.policy
        _write_run(args.output, world, summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset":
        summary = generate_sft_dataset(
            output_path=args.output,
            world_config=config.world,
            episodes=config.dataset.episodes if args.episodes is None else args.episodes,
            steps_per_episode=(
                config.dataset.steps_per_episode if args.steps is None else args.steps
            ),
            seed=config.dataset.seed if args.seed is None else args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset-balanced":
        summary = generate_balanced_sft_dataset(
            output_path=args.output,
            base_dataset_path=args.base,
            world_config=config.world,
            episodes=args.episodes,
            steps_per_episode=(
                config.dataset.steps_per_episode if args.steps is None else args.steps
            ),
            seed=args.seed,
            minimum_per_action=args.minimum_per_action,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset-stratify":
        summary = stratify_sft_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            records_per_action=args.per_action,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset-communication":
        summary = generate_communication_sft_dataset(
            output_path=args.output,
            base_dataset_path=args.base,
            world_config=config.world,
            scenes=args.scenes,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset-v4":
        summary = generate_v4_sft_dataset(
            output_path=args.output,
            base_dataset_path=args.base,
            world_config=config.world,
            scenes=args.scenes,
            survival_scenes=args.survival_scenes,
            seed=args.seed,
            include_trajectories=not args.no_trajectories,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "dataset-v41":
        summary = generate_v41_sft_dataset(
            output_path=args.output,
            base_dataset_path=args.base,
            world_config=config.world,
            rich_scenes=args.rich_scenes,
            reachability_scenes=args.reachability_scenes,
            seed=args.seed,
            include_trajectories=not args.no_trajectories,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "audit-v41":
        summary = audit_v41_datasets(args.training, args.held_out)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return 0 if summary["all_checks_pass"] else 1

    if args.command == "train":
        from .training import train_qlora_sft

        training_config = config.training
        overrides = {
            "model": args.model,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        training_config = replace(
            training_config,
            **{key: value for key, value in overrides.items() if value is not None},
        )
        manifest = train_qlora_sft(
            dataset_path=args.dataset,
            output_dir=args.output,
            config=training_config,
            seed=config.run.seed if args.seed is None else args.seed,
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    if args.command == "run-model":
        from .model_policy import QwenPolicy

        seed = config.run.seed if args.seed is None else args.seed
        world_config = replace(
            config.world,
            founders=args.founders,
            max_population=max(args.founders, min(config.world.max_population, args.founders * 3)),
        )
        policy = QwenPolicy(
            model_name=args.model or config.training.model,
            adapter_path=args.adapter,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
        world = World(world_config, seed=seed)
        summary = world.run(policy, args.steps)
        summary.update({"policy": "model", "model": args.model or config.training.model})
        if args.adapter:
            summary["adapter"] = str(Path(args.adapter).resolve())
        _write_run(args.output, world, summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run-communication-world":
        from .communication import run_communication_experiment
        model_name = args.model or config.training.model
        policy = None
        policy_factory = None
        if args.policy == "model":
            from .model_policy import QwenPolicy

            policy = QwenPolicy(
                model_name=model_name,
                adapter_path=args.adapter,
                temperature=args.temperature,
                load_in_4bit=args.load_in_4bit,
            )
        else:
            policy_factory = (
                lambda scene_seed, _condition: HeuristicPolicy(scene_seed + 100)
            )
        summary = run_communication_experiment(
            policy=policy,
            policy_factory=policy_factory,
            output_dir=args.output,
            seeds=range(args.seed_start, args.seed_start + args.scenes),
            steps=args.steps,
            profile=args.profile,
            metadata={
                "policy": args.policy,
                "model": model_name if args.policy == "model" else None,
                "adapter": (
                    str(Path(args.adapter).resolve())
                    if args.adapter and args.policy == "model"
                    else None
                ),
                "temperature": args.temperature,
            },
        )
        console_summary = {
            key: value for key, value in summary.items() if key != "runs"
        }
        console_summary["summary_path"] = str(
            (Path(args.output) / "summary.json").resolve()
        )
        print(json.dumps(console_summary, indent=2))
        return 0

    if args.command == "evaluate":
        from .evaluation import evaluate_policy
        from .model_policy import QwenPolicy

        policy = QwenPolicy(
            model_name=args.model or config.training.model,
            adapter_path=args.adapter,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
        summary = evaluate_policy(
            policy=policy,
            dataset_path=args.dataset,
            output_path=args.output,
            max_records=args.max_records,
            seed=args.seed,
        )
        summary_path = Path(args.output).with_name("summary.json")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "evaluate-communication":
        from .evaluation import evaluate_communication_policy
        from .model_policy import QwenPolicy

        policy = QwenPolicy(
            model_name=args.model or config.training.model,
            adapter_path=args.adapter,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
        summary = evaluate_communication_policy(
            policy=policy,
            dataset_path=args.dataset,
            output_path=args.output,
        )
        summary_path = Path(args.output).with_name("summary.json")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
        return 0

    return 2


def doctor() -> int:
    report: dict[str, Any] = {"python": sys.version, "executable": sys.executable}
    packages = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate")
    report["packages"] = {}
    for package in packages:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = "missing"

    try:
        import torch

        report["torch_cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            report["gpu"] = torch.cuda.get_device_name(device)
            report["compute_capability"] = list(torch.cuda.get_device_capability(device))
            report["bf16_supported"] = torch.cuda.is_bf16_supported()
            left = torch.randn((32, 32), device=device, dtype=torch.bfloat16)
            right = torch.randn((32, 32), device=device, dtype=torch.bfloat16)
            report["bf16_matmul_finite"] = bool(torch.isfinite(left @ right).all().item())
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import bitsandbytes  # noqa: F401

        report["bitsandbytes_import"] = True
    except Exception as exc:
        report["bitsandbytes_import"] = False
        report["bitsandbytes_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(report, indent=2))
    required_ok = (
        report.get("cuda_available") is True
        and report.get("bf16_supported") is True
        and report.get("bitsandbytes_import") is True
    )
    return 0 if required_ok else 1


def _write_run(output_dir: str | Path, world: World, summary: dict[str, Any]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    world.write_events(destination / "events.jsonl")
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

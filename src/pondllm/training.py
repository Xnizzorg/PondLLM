from __future__ import annotations

import inspect
import json
from importlib import metadata
from pathlib import Path

from .config import TrainingConfig


def train_qlora_sft(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    seed: int = 7,
) -> dict[str, object]:
    _stage("Importing the GPU training stack")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; install the 'train' extra") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this configuration requires BF16 support")

    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _stage(f"Loading SFT dataset from {source}")
    dataset = load_dataset("json", data_files=str(source), split="train")
    if len(dataset) < 20:
        raise ValueError("at least 20 SFT records are required")
    split = dataset.train_test_split(test_size=min(0.1, 200 / len(dataset)), seed=seed)

    _stage(f"Loading tokenizer for {config.model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    _stage(f"Loading the 4-bit base model {config.model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    requested_args = {
        "output_dir": str(destination),
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.micro_batch_size,
        "per_device_eval_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "max_length": config.max_length,
        "completion_only_loss": True,
        "dataset_num_proc": 1,
    }
    accepted = inspect.signature(SFTConfig).parameters
    training_args = SFTConfig(**{key: value for key, value in requested_args.items() if key in accepted})

    _stage("Constructing the SFT trainer")
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "peft_config": peft_config,
    }
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    _stage("Starting optimizer steps")
    result = trainer.train()
    _stage(f"Saving the trained adapter to {destination}")
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    manifest = {
        "model": config.model,
        "dataset": str(source.resolve()),
        "dataset_records": len(dataset),
        "train_records": len(split["train"]),
        "eval_records": len(split["test"]),
        "seed": seed,
        "training": {
            "max_length": config.max_length,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        },
        "versions": _package_versions(),
        "metrics": result.metrics,
    }
    with (destination / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest


def _package_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _stage(message: str) -> None:
    print(f"[pondllm] {message}", flush=True)

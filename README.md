# PondLLM

PondLLM is a small artificial ecology in which organisms share a frozen language model while
lineages can inherit different LoRA adapters. Individual memory, heritable controller genes,
and adapter evolution are deliberately kept separate.

The first milestone is intentionally modest: train one adapter to map a bounded local observation
to one legal JSON action, then compare it with unadapted and heuristic policies on held-out worlds.

## Current scope

- Deterministic grid ecology with local perception
- Energy, food, movement, sharing, signalling, rest, reproduction, and death
- Heritable scalar genes with bounded mutation
- Structured individual memory
- JSONL trajectory and ancestry logs
- Heuristic demonstrations for SFT
- Optional Qwen inference and QLoRA training
- No RL or raw adapter-weight mutation yet

## Quick start

Activate the Python environment containing the installed packages, then install this repository in
editable mode:

```powershell
python -m pip install -e .
python -m pondllm doctor
python -m pondllm smoke --steps 40
python -m unittest discover -s tests -v
```

The simulator and dataset generator use only the Python standard library. The `doctor`, `train`, and
`run-model` commands import the GPU stack lazily.

Generate supervised examples:

```powershell
python -m pondllm dataset --episodes 16 --steps 60 --seed 1000 --output data/generated/sft.jsonl
python -m pondllm dataset --episodes 8 --steps 60 --seed 9000 --output data/generated/eval.jsonl
```

Train the first adapter:

```powershell
python -m pondllm train `
  --dataset data/generated/sft.jsonl `
  --output artifacts/qwen3-0.6b-action-sft
```

For a quick kernel/API probe before the full run, use a small dataset and fractional epoch:

```powershell
python -m pondllm train `
  --dataset data/generated/sft-smoke.jsonl `
  --output artifacts/qlora-probe `
  --epochs 0.1
```

Evaluate JSON syntax and observation-level legality before ecological evaluation:

```powershell
python -m pondllm evaluate `
  --dataset data/generated/eval.jsonl `
  --adapter artifacts/qwen3-0.6b-action-sft
```

The initial gate is at least 99% parseable JSON and 95% observation-legal actions on held-out
states. Passing that gate does not establish ecological fitness; it only makes a population run
worth interpreting.

Run a small model-controlled population:

```powershell
python -m pondllm run-model `
  --adapter artifacts/qwen3-0.6b-action-sft `
  --steps 30 `
  --output runs/model-smoke
```

Model downloads are cached by Hugging Face and are not committed to this repository.

## Balanced-action V2

The second adapter keeps the V1 setup fixed while topping up unique `reproduce`, `share`, and
`signal` demonstrations to 1,000 records each:

```powershell
python -m pondllm dataset-balanced `
  --base data/generated/sft.jsonl `
  --episodes 128 `
  --seed 2000 `
  --minimum-per-action 1000 `
  --output data/generated/sft-v2-balanced.jsonl

python -m pondllm dataset --episodes 32 --steps 60 --seed 9000 `
  --output data/generated/eval-v2-pool.jsonl

python -m pondllm dataset-stratify `
  --dataset data/generated/eval-v2-pool.jsonl `
  --per-action 100 `
  --seed 23 `
  --output data/generated/eval-v2-stratified.jsonl
```

V2 achieved 600/600 parseable and 599/600 legal held-out actions. It learned reproduction strongly,
sharing partially, and signalling only weakly. The full controlled comparison is recorded in
[docs/night-two-results.md](docs/night-two-results.md).

The first experiment remains available in [docs/night-one-results.md](docs/night-one-results.md).

## Qwen3.5-0.8B-Base transplantation

The third experiment trained the unchanged balanced dataset and QLoRA configuration against the
pretrained-only `Qwen/Qwen3.5-0.8B-Base` checkpoint:

```powershell
python -m pondllm train `
  --dataset data/generated/sft-v2-balanced.jsonl `
  --output artifacts/qwen3.5-0.8b-base-action-sft-v1-balanced `
  --model Qwen/Qwen3.5-0.8B-Base
```

On the same 600 stratified cases, it achieved 600/600 parseable and 599/600 legal actions, with
70.33% macro action accuracy. Sharing recall improved from 44% to 85% relative to the Qwen3-0.6B
V2 adapter, but signalling remained unlearned. The controlled comparison and four-world ecology
results are in [docs/night-three-results.md](docs/night-three-results.md). The adapter is published
at [Xnizzorg/pondllm-qwen3.5-0.8b-base-action-sft](https://huggingface.co/Xnizzorg/pondllm-qwen3.5-0.8b-base-action-sft).

## Causal communication V3

The fourth experiment adds paired sender and recipient cases: signal useful food coordinates,
suppress redundant signals, act on received information, and preserve the no-memory control.

Both fresh Qwen3.5 adapters passed the disjoint 600-case communication gate. The 0.8B model reached
96.67% useful signalling and 99.33% recipient-informed accuracy; the 2B model reached 99.33% and
100.00%, respectively. Both produced 0% redundant signals and 100% legal actions.

The full corpus design, predetermined gates, legacy-action checks, fixed-seed world runs, caveats,
hashes, and 0.8B/2B comparison are in
[docs/night-four-communication-results.md](docs/night-four-communication-results.md).

The public V3 adapters are:

- [Qwen3.5-0.8B-Base communication adapter](https://huggingface.co/Xnizzorg/pondllm-qwen3.5-0.8b-base-communication-sft)
- [Qwen3.5-2B-Base communication adapter](https://huggingface.co/Xnizzorg/pondllm-qwen3.5-2b-base-communication-sft)

## World-level communication ablation

The fifth experiment put both adapters into paired live worlds with normal, blocked, corrupted,
and costly signal channels. With training-shaped sender/recipient context, signalling caused food
acquisition in 3/4 scenes for 0.8B and 4/4 for 2B; blocking delivery reduced both to 0/4.

That behavior did not generalize. With neutral organism and lineage IDs, tick zero, and empty
memories, neither adapter signalled in any of four scenes. The earlier held-out score measured
seed generalization inside one generator schema, not robust recognition of information asymmetry.
The result, caveats, free-world observations, and V4 curriculum are recorded in
[docs/night-five-world-communication.md](docs/night-five-world-communication.md).

Reproduce a neutral-profile run with:

```powershell
python -m pondllm run-communication-world `
  --model Qwen/Qwen3.5-0.8B-Base `
  --adapter artifacts/qwen3.5-0.8b-base-action-sft-v3-communication `
  --profile clean `
  --scenes 4 `
  --steps 7 `
  --temperature 0 `
  --output runs/communication-world-0.8b-clean
```

## Repository map

```text
configs/night_one.toml       Night-one defaults
src/pondllm/domain.py        Actions, organisms, genes, and observations
src/pondllm/world.py         Deterministic ecology and event logging
src/pondllm/communication.py Paired world-level signal interventions and metrics
src/pondllm/policies.py      Heuristic and random baselines
src/pondllm/prompting.py     Shared action and communication protocol
src/pondllm/model_policy.py  Optional local Qwen policy
src/pondllm/dataset.py       SFT demonstration generation
src/pondllm/evaluation.py    Held-out action and causal communication metrics
src/pondllm/training.py      QLoRA SFT entry point
src/pondllm/cli.py           Command-line interface
tests/                       Simulator and parser tests
```

## Experimental discipline

Always preserve the seed, model revision, adapter ancestry, configuration, and full event stream.
An apparent improvement is not useful unless it survives held-out seeds and environmental changes.

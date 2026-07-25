# Night-two results: balanced action competence

Date: 2026-07-23/24  
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB  
Base model: `Qwen/Qwen3-0.6B`  
Method: fresh NF4 QLoRA SFT, rank 8, alpha 16, two epochs

## Question

The first PondLLM adapter produced legal actions but collapsed onto `move`, `forage`, and `rest`.
Night two tested a single controlled change: increase unique training coverage of the rare
`reproduce`, `share`, and `signal` actions while keeping the model, prompt, optimizer, LoRA
configuration, seed, and simulator unchanged.

This is an action-competence experiment, not reward shaping or an ecological-fitness claim.

## Deterministic data

The original natural dataset contained 11,614 records. One exact duplicate was removed before
balancing. New demonstrations came from simulator seeds 2000–2100, disjoint from both the original
training seeds and evaluation seeds. Candidate generation stopped as soon as every rare action
reached 1,000 unique examples.

| Action | V1 records | V2 records |
|---|---:|---:|
| Forage | 2,463 | 2,463 |
| Move | 6,495 | 6,495 |
| Reproduce | 137 | 1,000 |
| Rest | 2,049 | 2,048 |
| Share | 154 | 1,000 |
| Signal | 316 | 1,000 |
| **Total** | **11,614** | **14,006** |

The held-out pool used seeds 9000–9031. A deterministic stratified sample selected 100 examples of
each action, for 600 evaluation records total.

## Training

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Held-out completion loss | 0.04628 | 0.04521 |
| Held-out token accuracy | 97.92% | 97.73% |

Training completed 864 optimizer steps in 35 minutes 44 seconds. Aggregate training loss was
0.07123. The final learned adapter is 10,143,888 bytes.

## Stratified held-out evaluation

All policies used greedy decoding on the same 600 records in the same deterministic order.

| Policy | Parseable | Legal | Macro action accuracy | Exact action accuracy |
|---|---:|---:|---:|---:|
| V2 balanced adapter | 100.00% | 99.83% | 67.33% | 53.17% |
| V1 natural adapter | 100.00% | 100.00% | 45.83% | 43.00% |
| Unadapted base | 100.00% | 17.67% | 16.00% | 15.50% |

Per-action recall:

| Expected action | V2 | V1 | Unadapted |
|---|---:|---:|---:|
| Forage | 100% | 100% | 90% |
| Move | 65% | 75% | 4% |
| Reproduce | 98% | 0% | 0% |
| Rest | 96% | 100% | 2% |
| Share | 44% | 0% | 0% |
| Signal | 1% | 0% | 0% |

V2 emitted all six action kinds. Its one illegal prediction was a `share` action from an organism
with one unit of energy, which would have spent its final unit. There were no parsing failures.

Balancing raised macro action accuracy by 21.5 percentage points over V1. It solved reproduction
competence and partially learned sharing. Signalling remains effectively unresolved.

## Fixed-seed live ecology

Each policy ran seeds 7–10 with four founders, 40 ticks, and greedy decoding. Every model output,
transition, birth, death, invalid action, and policy failure is retained in the corresponding
`events.jsonl`.

| Policy | Decisions | Valid | Errors | Births | Deaths | Living | Surviving lineage instances | Mean final energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 balanced adapter | 832 | 832 | 0 | 12 | 3 | 25 | 16/16 | 8.107 |
| V1 natural adapter | 640 | 640 | 0 | 0 | 0 | 16 | 16/16 | 19.875 |
| Heuristic | 764 | 764 | 0 | 11 | 7 | 20 | 16/16 | 7.275 |
| Unadapted base | 172 | 3 | 0 | 0 | 16 | 0 | 0/16 | 0.000 |

V2 used every rare action in live ecology except that signalling appeared only once:

```text
forage 142
move 285
reproduce 12
rest 355
share 37
signal 1
```

Every V2 live decision was valid. Each of its 12 reproduction actions created an offspring. All
founder lineages survived every evaluated world.

This four-seed result demonstrates a behavioural change, not general ecological superiority. V1
ended with higher mean energy because it never paid reproduction costs. Larger seed suites and
environmental regime changes are required before comparing fitness.

## Reproducibility artifacts

| Artifact | SHA-256 |
|---|---|
| `adapter_model.safetensors` | `64A24413099F71DEBC10AEE5CF863EC9A83C51C3B9EE8F4F1BDB76C96D3308DB` |
| `run_manifest.json` | `8A9E22316F673B91C264869A722EBBAA39DE462A61D562B84E7D047A38B403AC` |
| `sft-v2-balanced.jsonl` | `C1F2569F080450D14B8BBD9674CEA691FCD9AF80D7CAD42BF442BD832F331817` |
| `eval-v2-stratified.jsonl` | `578EF1EB4B45B03104BFDAF3C32B347C9AA3504CB607181C5113B5BCA6BA431C` |

Primary locations:

```text
artifacts/qwen3-0.6b-action-sft-v2-balanced/
data/generated/sft-v2-balanced.jsonl
data/generated/sft-v2-balanced.summary.json
data/generated/eval-v2-pool.jsonl
data/generated/eval-v2-stratified.jsonl
data/generated/eval-v2-stratified.summary.json
runs/v2-stratified-eval/
runs/v1-stratified-eval/
runs/base-stratified-eval/
runs/fixed-seed-comparison/summary.json
```

## Next supervised experiment

Do not move to population-scale RL yet. The next narrow experiment should target conditional
communication: generate diverse signalling situations and messages, evaluate whether message
content matches observable state, and preserve the reproduction and sharing gains from V2.

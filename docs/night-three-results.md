# Night-three results: Qwen3.5-0.8B-Base transplantation

Date: 2026-07-24/25  
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB  
New base model: `Qwen/Qwen3.5-0.8B-Base`  
Method: fresh NF4 QLoRA SFT, rank 8, alpha 16, two epochs

## Question

Night three transplanted the existing balanced PondLLM action task from the post-trained
`Qwen/Qwen3-0.6B` checkpoint to the pretrained-only `Qwen/Qwen3.5-0.8B-Base` checkpoint.

The dataset, prompt, seed, LoRA rank and alpha, learning rate, effective batch size, epoch count,
held-out cases, decoding method, simulator, and live-world seeds were held fixed.

The narrow question was:

> Does the same supervised behavior transfer to Qwen3.5-0.8B-Base, and does that checkpoint produce
> a better bounded action policy without changing the demonstrations?

This is a controlled transplantation, but not a pure architecture ablation. The two checkpoints
also differ in tokenizer, vocabulary, pretraining, and post-training stage.

## Compatibility

Transformers 5.14 loaded the multimodal Qwen3.5 repository through its text-only
`Qwen3_5ForCausalLM` path. The LoRA targeted all linear language-model projections:

- 18 Gated DeltaNet blocks
- 6 conventional gated-attention blocks
- 24 feed-forward blocks

The vision encoder and tied token embeddings were not adapted.

The optional fused DeltaNet and causal-convolution kernels were unavailable in the Windows
environment, so training and inference used the supported PyTorch fallback.

## Training

The same 14,006-record balanced dataset from night two was used. It contains 1,000 examples each
of `reproduce`, `share`, and `signal`, with natural frequencies retained for the common actions.

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Held-out completion loss | 0.06049 | 0.06031 |
| Held-out token accuracy | 97.18% | 97.20% |

Training completed 864 optimizer steps in 75 minutes 50 seconds. Aggregate training loss was
0.07775. The final adapter is 10,872,304 bytes.

The same Qwen3-0.6B V2 run took 35 minutes 44 seconds. Most of the Qwen3.5 slowdown came from the
unfused Windows fallback rather than memory pressure.

## Stratified held-out evaluation

Every policy used greedy decoding on the same 600 observations: 100 examples of each action.

| Policy | Parseable | Legal | Macro action accuracy | Exact action accuracy |
|---|---:|---:|---:|---:|
| **Qwen3.5-0.8B-Base adapter** | **100.00%** | **99.83%** | **70.33%** | **58.00%** |
| Qwen3-0.6B V2 adapter | 100.00% | 99.83% | 67.33% | 53.17% |
| Qwen3.5-0.8B-Base unadapted | 100.00% | 2.00% | 16.67% | 0.83% |
| Qwen3-0.6B unadapted | 100.00% | 17.67% | 16.00% | 15.50% |

Per-action recall:

| Expected action | Qwen3.5 adapter | Qwen3 V2 adapter | Difference |
|---|---:|---:|---:|
| Forage | 100% | 100% | 0 pp |
| Move | 62% | 65% | -3 pp |
| Reproduce | 90% | 98% | -8 pp |
| Rest | 85% | 96% | -11 pp |
| Share | 85% | 44% | +41 pp |
| Signal | 0% | 1% | -1 pp |

The three-point macro improvement came from a large sharing gain, partly offset by weaker
reproduction, rest, and movement recall. The substrate change did not solve signalling.

The adapter emitted five action kinds and never emitted `signal`. Its one illegal held-out
prediction tried to share the final unit of energy. The unadapted Qwen3.5 base emitted
`reproduce` for all 600 cases; only 12 of those attempts were legal.

## Fixed-seed live ecology

Each policy ran seeds 7–10 with four founders, 40 ticks, and greedy decoding. The Qwen3 and
heuristic results are carried forward from the unchanged night-two fixed-seed suite.

| Policy | Decisions | Valid | Births | Deaths | Living | Surviving lineage instances | Mean final energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.5 adapter** | **841** | **840** | **13** | **4** | **25** | **16/16** | **8.840** |
| Qwen3 V2 adapter | 832 | 832 | 12 | 3 | 25 | 16/16 | 8.107 |
| Heuristic | 764 | 764 | 11 | 7 | 20 | 16/16 | 7.275 |
| Qwen3.5 unadapted | 160 | 0 | 0 | 16 | 0 | 0/16 | 0.000 |

Qwen3.5 adapted action counts:

```text
forage 150
move 312
reproduce 14
rest 354
share 11
signal 0
```

Thirteen of 14 reproduction attempts created offspring. The one invalid action came from an
organism with energy 16 whose inherited reproduction threshold had mutated to 17. Every founder
lineage survived every adapted world.

The unadapted model attempted reproduction on all 160 live decisions, never met the threshold,
and drove every founder to death by tick 10.

Decision counts differ because births and deaths change the number of organisms acting in later
ticks. These four worlds demonstrate protocol learning and behavioral transplantation, not broad
ecological superiority.

## Interpretation

Qwen3.5-0.8B-Base is a viable Pond substrate. A small LoRA taught a pretrained-only model a strict
JSON action protocol, raised observation-level legality from 2.00% to 99.83%, preserved all
lineages in live worlds, and transferred all useful action classes except communication.

It is not an unqualified replacement for Qwen3-0.6B:

- It improved overall action agreement and sharing substantially.
- It was weaker on reproduction, rest, and movement.
- It was roughly twice as slow to train in this Windows environment.
- It did not learn signalling at all.

The shared signal failure across both model families is evidence that the next bottleneck is the
training target, not simply parameter count or architecture. Signal demonstrations currently
reward emitting a template string, but they do not establish when communication changes another
organism's future behavior.

## Reproducibility artifacts

| Artifact | SHA-256 |
|---|---|
| `adapter_model.safetensors` | `90FF3DD4615059E16EED9FC46698F47AEA259A7F97171BCF6A5578B14619160D` |
| `run_manifest.json` | `42FE7C2C21BBF5420C3E48EA20F3DCE7F482FFDA430AFD5837065075F4D33A99` |
| `sft-v2-balanced.jsonl` | `C1F2569F080450D14B8BBD9674CEA691FCD9AF80D7CAD42BF442BD832F331817` |
| `eval-v2-stratified.jsonl` | `578EF1EB4B45B03104BFDAF3C32B347C9AA3504CB607181C5113B5BCA6BA431C` |

Primary locations:

```text
artifacts/qwen3.5-0.8b-base-action-sft-v1-balanced/
runs/qwen35-base-stratified-eval/
runs/qwen35-unadapted-stratified-eval/
runs/fixed-seed-qwen35-adapted/
runs/fixed-seed-qwen35-unadapted/
runs/fixed-seed-qwen35-comparison/summary.json
```

## Next supervised experiment

Keep the current action adapter as the control. Build communication trajectories in which a
signal contains locally observable information, is received by another organism, and causes a
measurable later action. Evaluate semantic accuracy, recipient response, and lineage outcome
separately. Do not begin population-scale RL until that supervised communication gate is defined.

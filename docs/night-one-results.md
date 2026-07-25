# Night-one results

Date: 2026-07-22/23  
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB  
Model: `Qwen/Qwen3-0.6B`  
Method: NF4 QLoRA, rank 8, alpha 16, two epochs

## Environment

```text
Python          3.13.3
PyTorch         2.11.0+cu130
Transformers    5.14.1
TRL             1.9.0
PEFT            0.19.1
bitsandbytes    0.49.2
CUDA available  yes
BF16 supported  yes
Compute cap.    12.0
```

The measured throughput optimum was micro-batch 16 with gradient accumulation 2, preserving an
effective batch size of 32. Sustained training used approximately 5.5 GB of GPU memory and completed
714 optimizer steps in 28 minutes 20 seconds.

## Data

The dependency-free simulator generated demonstrations from seeds disjoint from evaluation:

| Split | Seeds | Records |
|---|---:|---:|
| Train | 1000–1015 | 11,614 |
| Behavioral evaluation | 9000–9007 | 5,763 available; 200 sampled |

All six action types occur in the training data. Demonstrations intentionally increase coverage of
rare legal `share` and `signal` actions while the evaluation heuristic remains unchanged.

## Training

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Held-out completion loss | 0.04160 | 0.04093 |
| Held-out token accuracy | 98.23% | 97.99% |

Final aggregate training loss was 0.06291. Gradient norms stayed controlled and no CUDA, OOM,
quantization, masking, or checkpoint errors occurred.

## Behavioral evaluation

The unadapted model's initial eight-state probe emitted valid JSON but chose `forage` for every
state, producing only 12.5% legal actions. This small probe is diagnostic rather than a statistically
matched baseline.

On 200 unseen states, the trained adapter achieved:

| Metric | Result |
|---|---:|
| Parseable action objects | 200/200 (100%) |
| Observation-legal actions | 200/200 (100%) |
| Demonstration action-kind agreement | 165/200 (82.5%) |
| Exact action agreement | 156/200 (78.0%) |

The expected sample contained 45 forage, 105 move, 40 rest, 7 share, and 3 signal examples. Greedy
adapter output contained 45 forage, 89 move, and 66 rest actions. It learned the core survival policy
but conservatively replaced rare social actions with movement or rest. Rare-action balancing is the
next supervised-learning experiment; it should not be confused with ecological fitness shaping.

## First live ecology

Identical seed-7 worlds used four founders and a 30-step limit:

| Policy | Ticks | Births | Deaths | Living | Mean energy |
|---|---:|---:|---:|---:|---:|
| Trained adapter, greedy | 30 | 0 | 0 | 4 | 17.0 |
| Heuristic | 30 | 1 | 1 | 4 | 10.5 |
| Random | 14 | 0 | 4 | 0 | 0.0 |

The adapter produced 120/120 legal live decisions with zero policy errors. It used 28 forage,
56 move, and 36 rest actions. All four founder lineages survived, but no reproduction occurred.

This result validates the local training and ecology pipeline. It is not yet an evolutionary result:
the next milestone is lineage-specific adapter inheritance, balanced action competence, and repeated
fixed-seed comparisons across environmental regimes.


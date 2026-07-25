# Night-four results: causal communication SFT

Date: 2026-07-24/25  
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB  
Models: `Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-2B-Base`  
Method: fresh NF4 QLoRA SFT, rank 8, alpha 16, two epochs

## Question

The earlier action curriculum taught a valid Pond action protocol but did not teach communication:
the 0.8B V1 adapter emitted no useful signals on a held-out causal test.

Night four tested a narrower claim:

> Can a small model learn to signal locally observed food only when another organism needs the
> information, and can a recipient change its action when that signal appears in memory?

This is supervised causal discrimination, not emergent language. The signal vocabulary and
teacher policy are specified by the dataset.

## V3 corpus

The 14,006-record V2 action corpus was retained and combined with 6,000 deterministic
communication records, yielding 20,006 unique examples. Each of 1,500 generated scenes contributes
four cases:

1. a sender signals a locally visible food coordinate that a nearby recipient cannot see;
2. the sender does not signal when the recipient already sees that food;
3. an informed recipient moves toward food named in a received signal;
4. the otherwise identical control recipient, without the memory, follows its local policy.

The held-out communication set contains 600 records (150 per case) generated from a disjoint seed.
All 300 informed/control pairs are complete and differ only in memory. There is no exact
prompt/completion overlap between the training and communication evaluation sets.

The action system prompt now documents signal delivery, one-step recipient memory, the canonical
`food at [x,y]` payload, and the organism's perception radius.

## Training

Both models used the same records, split seed, two epochs, effective batch size 32, maximum length
1,024, learning rate `2e-4`, LoRA rank 8, alpha 16, and dropout 0.05.

| Model | Micro batch | Accumulation | Steps | Runtime | Train loss | Final eval loss | Eval token accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-Base | 16 | 2 | 1,238 | 2:03:04 | 0.05906 | 0.0385 | 98.11% |
| Qwen3.5-2B-Base | 4 | 8 | 1,238 | 3:38:39 | 0.05582 | 0.03666 | 98.26% |

The 2B configuration was first checked with a one-step VRAM probe. Training then ran once; the
overnight monitor inspected and resumed the existing process rather than starting a duplicate.
The optional fused Qwen3.5 kernels were unavailable on Windows, so both models used the supported
PyTorch fallback.

## Held-out communication gate

The predetermined gate required:

- syntax and legality at least 99%;
- useful-signal rate at least 70%;
- redundant-signal rate at most 10%;
- payload-coordinate accuracy at least 70%;
- recipient-informed and recipient-control exact accuracy at least 70%;
- preservation of common legacy action behavior.

Greedy decoding results on the 600 held-out cases:

| Measure | Old 0.8B V1 adapter | 0.8B V3 | 2B V3 | Gate |
|---|---:|---:|---:|---:|
| Syntax valid | 100.00% | 100.00% | 100.00% | >=99% |
| Legal | 100.00% | 100.00% | 100.00% | >=99% |
| Useful signal | 0.00% | 96.67% | 99.33% | >=70% |
| Redundant signal | 0.00% | 0.00% | 0.00% | <=10% |
| Correct payload coordinate | 0.00% | 96.67% | 99.33% | >=70% |
| Recipient informed exact | 0.00% | 99.33% | 100.00% | >=70% |
| Recipient control exact | 91.33% | 100.00% | 100.00% | >=70% |
| Complete-pair exact | 0.00% | 98.00% | 99.00% | diagnostic |

Both V3 adapters pass every explicit communication threshold. The 2B model is slightly more
accurate, but the 0.8B model already clears the gate by a wide margin.

The old adapter's 0% redundant-signal rate should not be read as good communication: it never
signalled in either useful or redundant cases.

## Legacy fixed-stratified action check

The unchanged V2 evaluation has 600 examples, 100 per expected action.

| Policy | Syntax | Legal | Macro action accuracy | Exact accuracy |
|---|---:|---:|---:|---:|
| 0.8B V1 action adapter | 100.00% | 99.83% | 70.33% | 58.00% |
| 0.8B V3 communication adapter | 100.00% | 99.83% | 66.33% | 51.83% |
| 2B V3 communication adapter | 100.00% | 100.00% | 70.50% | 58.17% |

Per-action recall:

| Expected action | 0.8B V1 | 0.8B V3 | 2B V3 |
|---|---:|---:|---:|
| Forage | 100% | 100% | 100% |
| Move | 62% | 70% | 64% |
| Reproduce | 90% | 92% | 86% |
| Rest | 85% | 87% | 92% |
| Share | 85% | 49% | 81% |
| Signal | 0% | 0% | 0% |

The 0.8B V3 model preserved or improved the four common deterministic action classes, but its
aggregate score fell because sharing weakened. The 2B V3 model recovered the earlier aggregate
level and retained strong sharing.

The legacy `signal` score remains zero for both V3 models because that old class rewards a
generic, randomly selected template payload. The V3 curriculum instead requires a useful food
coordinate and suppresses signalling when it is redundant. The causal V3 evaluation is therefore
the meaningful signal measure. Some V2 `share` targets also contain hidden teacher randomness, so
exact agreement is not equivalent to ecological correctness.

## Fixed-seed live-world check

Each V3 model ran seed 7 for 30 ticks with four founders and greedy decoding.

| Model | Decisions | Valid | Births | Deaths | Living | Founder lineages surviving | Mean final energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B V3 | 151 | 151 | 3 | 1 | 6 | 4/4 | 6.167 |
| 2B V3 | 151 | 151 | 2 | 0 | 6 | 4/4 | 8.833 |

Action counts:

| Model | Forage | Move | Reproduce | Rest | Share | Signal |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B V3 | 29 | 73 | 3 | 44 | 2 | 0 |
| 2B V3 | 26 | 52 | 2 | 68 | 3 | 0 |

There were no parse failures or illegal actions. No live signal opportunity occurred under this
single seed, so the world run checks protocol stability and viability, not communication efficacy.
The paired held-out suite remains the causal communication test.

## Interpretation

The main result is that signalling failure was a curriculum problem, not a hard limitation of the
0.8B architecture. Adding paired sender and recipient examples changed the 0.8B model from never
using communication to 96.67% useful signalling and 99.33% exact recipient response while
preserving ordinary forage, movement, reproduction, and rest behavior.

The 2B run improved the communication result by only a few percentage points but recovered the
legacy aggregate action score and sharing behavior. It is also about 1.78 times slower to train and
about 1.26 times slower per evaluation record in this Windows fallback environment. For broad
iteration, 0.8B remains the economical research model; 2B is the stronger reference when ordinary
action retention matters.

This experiment does not show emergent communication, language invention, or ecological benefit.
It shows that the model can learn a specified information-transfer policy with three linked
properties: selective sending, accurate content, and recipient action change.

The next experiment should place genuine information asymmetry into fixed-seed multi-agent worlds
and compare lineage outcomes with communication enabled, disabled, corrupted, and costly. The
current adapters are suitable supervised controls for that ablation. Population-scale RL remains
premature until the causal world-level evaluation exists.

## Reproducibility artifacts

| Artifact | SHA-256 |
|---|---|
| 0.8B `adapter_model.safetensors` | `9CA9034D83B9E2C517D47EEB53AE9D39479CD8523058DD2F6C4687AA4D225C64` |
| 2B `adapter_model.safetensors` | `2CFC17F1FB1E20EC7EBDFD00B76990ADF394C823B732C44AF26634F79CDD6102` |
| `sft-v3-communication.jsonl` | `DF664A0CEB4E755F417968CFD1112C380B9E7241F0D61F4C5B9363C770C7F760` |
| `eval-v3-communication.jsonl` | `0EF925618F6CA5956C6B782AC6BFCFDF20AF98B7D75CAC314111C5B15F2FF42A` |
| `eval-v2-stratified.jsonl` | `578EF1EB4B45B03104BFDAF3C32B347C9AA3504CB607181C5113B5BCA6BA431C` |

Primary locations:

```text
artifacts/qwen3.5-0.8b-base-action-sft-v3-communication/
artifacts/qwen3.5-2b-base-action-sft-v3-communication/
runs/qwen35-0.8b-v3-communication/
runs/qwen35-0.8b-v3-legacy-action/
runs/qwen35-0.8b-v3-live-seed-7/
runs/qwen35-2b-v3-communication/
runs/qwen35-2b-v3-legacy-action/
runs/qwen35-2b-v3-live-seed-7/
```

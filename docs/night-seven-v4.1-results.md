# Night-seven results: rich-context and reachability V4.1

Date: 2026-07-26
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB
Base model: `Qwen/Qwen3.5-0.8B-Base`
Method: fresh QLoRA SFT followed by greedy, disjoint, simulator-native evaluation

## Question

V4 passed controlled two-agent communication and survival gates, but it did not signal in 13
strict information-asymmetry opportunities across four ordinary worlds. Two low-energy offspring
also died while pursuing food that they could not reach before exhausting their energy.

V4.1 asks:

> Can a fresh 0.8B adapter recognize information asymmetry with four agents and two food patches,
> and can it condition movement and assistance on whether food is reachable before starvation?

This remains supervised learning on a frozen shared language model. It is not population-scale RL
or evidence of emergent language.

## Curriculum and harness

The existing deterministic simulator-native harness was reused. V4.1 adds:

- four-agent, two-food communication counterfactuals created from real `World.observe` snapshots;
- useful, redundant, current-food, informed-recipient, and no-message control cases;
- complete rich recipient trajectories;
- reachable-food, unreachable-food, and current-food counterfactuals;
- safe and unsafe parental rescue shares;
- rescued-child movement and unrescued-child waiting;
- a `v41` live profile with two forced-rest distractors and normal, blocked, corrupted, and costly
  signal channels;
- corpus audits for legality, duplicates, role-shaped identifiers, age/tick consistency,
  reachability labels, and prompt-plus-completion overlap.

The inherited 27,014-record V4 corpus is normalized to the current prompt. The final V4.1 training
set contains 43,322 records:

| Component | Records |
|---|---:|
| Normalized V4 corpus | 27,014 |
| Five rich communication cases | 5,000 |
| Rich recipient trajectories | 4,308 |
| Seven reachability and rescue cases | 7,000 |

The 1,000-record rich communication set and 1,050-record reachability set use disjoint seeds.
Audits found zero duplicates, illegal labels, role-shaped identifiers, age/tick errors,
reachability-label errors, or overlap with training.

The paired harness was also corrected for stochastic baselines. It can now construct a fresh policy
for every seed and channel condition, preventing heuristic RNG state from leaking between paired
interventions. A regression test covers this isolation.

## Gates

The gates used before evaluation were:

- at least 99% strict JSON and observation-legal actions;
- at least 70% useful rich-context signalling;
- no more than 10% redundant rich-context signalling;
- at least 70% correct signal coordinates;
- at least 70% exact current-food, informed-recipient, and no-message control behavior;
- at least 90% exact behavior in every reachability and rescue case;
- retention of the V4 controlled communication and survival behavior;
- at least a +50-point normal-versus-blocked forage effect in rich paired worlds, with lower
  success under corrupted coordinates;
- no population-scale RL if any controlled gate fails.

## Training

The adapter was trained fresh for two epochs with seed 7, 4-bit NF4 loading, BF16 compute, LoRA
rank 8 and alpha 16 over all linear layers, learning rate `2e-4`, micro-batch size 16, and two
gradient-accumulation steps. The effective batch size was 32 and loss was completion-only.

Training completed 2,696 optimizer steps in 22,753 seconds (6 hours 19 minutes). Aggregate training
loss was 0.02700. The fixed 200-record training split had 0.01901 loss and 98.94% mean token
accuracy after epoch one, improving to 0.01658 loss with 98.90% token accuracy after epoch two.

The manifest pins Git commit `340131a4f130218fff3d90a87bd1b9577a9e17d0` with
`git_dirty: false`.

## Results

### Rich communication

| Metric | Gate | V4 baseline | V4.1 |
|---|---:|---:|---:|
| Strict JSON | at least 99% | 100% | 100% |
| Observation-legal | at least 99% | 100% | 100% |
| Overall exact | — | 74.0% | 86.4% |
| Useful signal | at least 70% | 0% | 75.5% |
| Redundant signal | at most 10% | 0% | **18.0% (fail)** |
| Correct payload coordinate | at least 70% | 0% | 75.5% |
| Recipient informed exact | at least 70% | 96.5% | 100% |
| Recipient control exact | at least 70% | 74.0% | 74.5% |
| Sender currently on food exact | at least 70% | 100% | 100% |

V4.1 learns the missing rich-context behavior: useful signalling rises from 0% to 75.5%. It also
retains current-food priority and improves the informed recipient. The cost is over-signalling:
36 of 200 redundant cases produced a signal, so the 10% selectivity gate fails.

This is not generic forgetting. On the original 1,000-record V4 communication suite, V4.1 achieves
99% useful signalling, 6% redundant signalling, 99% payload accuracy, 99% informed-recipient
accuracy, 81% control accuracy, and 100% visible-food override accuracy. Every output is strict
JSON and legal.

### Reachability and rescue

| Case | Gate | V4 baseline | V4.1 |
|---|---:|---:|---:|
| Food current: forage | at least 90% | 100% | 100% |
| Food reachable: move | at least 90% | 87.33% | 96.67% |
| Food unreachable: rest | at least 90% | 0% | 91.33% |
| Safe rescue: share | at least 90% | 14% | 100% |
| Unsafe rescue: do not share | at least 90% | 100% | 100% |
| Rescued child: move | at least 90% | 100% | 100% |
| Unrescued child: wait | at least 90% | 0% | 100% |
| Overall exact | — | 57.33% | 98.29% |

V4.1 passes every static reachability and rescue gate. It makes a large controlled improvement over
V4, particularly on unreachable food, safe rescue sharing, and unrescued-child waiting.

The original V4 survival suite is also retained: V4.1 is 99.9% exact, with 100% safe sharing, 0%
unsafe-reserve sharing, 0% unnecessary sharing, and 100% low-energy food priority.

### Rich paired worlds

Sixteen layouts were run under normal, blocked, corrupted, and one-energy-cost channels.

| Condition | Sender signalled | Recipient foraged | Strict/legal model decisions |
|---|---:|---:|---:|
| Normal | 56.25% | 56.25% | 100% / 100% |
| Blocked | 56.25% | 0% | 100% / 100% |
| Corrupted | 56.25% | 0% | 100% / 100% |
| Costly | 56.25% | 56.25% | 100% / 100% |

The normal-minus-blocked forage effect is +56.25 percentage points. Normal recipients finish 2.188
Manhattan cells closer and with 1.562 more energy than blocked recipients on average. Corruption
removes all successful foraging. All nine costly signals are charged.

The reset-per-condition heuristic emits no signals and has an exactly zero paired effect. Its
recipient forages in 37.5% of every condition through local behavior. The unadapted base has 0%
strict outputs, 0% recipient foraging, and zero paired effect. A scripted positive control reaches
food in every normal V4.1 scene and none of the blocked scenes.

The adapter therefore has a real channel-dependent effect, but only nine of 16 rich senders choose
to communicate.

### Legacy action retention

All 600 outputs are strict JSON and observation-legal.

| Expected action | V4 recall | V4.1 recall |
|---|---:|---:|
| Forage | 100% | 100% |
| Move | 72% | 70% |
| Reproduce | 98% | 98% |
| Rest | 94% | 96% |
| Share | 0% | 4% |
| Generic signal | 0% | 0% |

The common actions are retained. Old generic signals and unconditional shares remain intentionally
in conflict with the resource-rational protocol and should not be treated as desired behavior.

### Ordinary fixed-seed worlds

The adapter was run for 30 ticks on seeds 7–10.

| Seed | Decisions | Births | Deaths | Living | Shares | Signals |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 142 | 2 | 1 | 5 | 0 | 1 |
| 8 | 138 | 2 | 0 | 6 | 0 | 0 |
| 9 | 140 | 2 | 2 | 4 | 2 | 1 |
| 10 | 154 | 3 | 0 | 7 | 0 | 1 |
| Total | 574 | 9 | 3 | 22 | 2 | 3 |

Every action is legal. V4.1 signals in three of 26 strict information-asymmetry opportunities
(11.54%), versus zero of 13 for V4. Each signal has at least one receiver with no current or
visible food, so they are not obviously redundant. This is partial spontaneous transfer, not yet
evidence that the signals improve fitness.

The two shares occur on consecutive ticks from an organism with 14 then 12 energy to a visible
one- and two-energy recipient. They satisfy the intended donor-safety pattern, but the recipient
later dies after moving at energy one.

All three deaths follow an energy-one move rather than a wait. Visible food is one or two cells
away, but reaching food also requires a later forage action. One death is a five-tick-old
offspring; two are founders aged 29 and 25. V4 had two deaths across these seeds, so the static
reachability improvement does not translate into a better four-world survival result.

## Interpretation and next step

V4.1 is a meaningful but incomplete correction:

- it changes rich useful signalling from 0% to 75.5%;
- it creates a +56.25-point causal forage effect in richer paired worlds;
- it changes static reachability/rescue exact accuracy from 57.33% to 98.29%;
- it retains V4 communication, survival, strict-action, and common legacy behavior.

It also fails the redundant-signal gate at 18%, signals in only nine of 16 rich live senders, and
does not improve ordinary-world deaths. Population-scale RL remains blocked.

A V4.2 SFT correction should:

1. mine the 36 rich redundant-signal errors as hard negatives and add recipient-specific partial
   visibility cases where some peers know the food and others do not;
2. add sequence-level energy-one trajectories with food one and two cells away, including
   distractors, prior signals, prior shares, founders, and offspring;
3. train the full move-then-forage energy budget, not only a static Manhattan-distance rule;
4. add a live rescue harness that measures donor reserve, repeated transfers, recipient waiting,
   eventual forage, and survival;
5. require improved ordinary-world survival and at most 10% rich redundant signalling before any
   population-scale RL.

## Reproducibility

| Artifact | SHA-256 |
|---|---|
| `sft-v4.1-rich-reachability.jsonl` | `84F5BAE00FCD2A1A98737E9A98371DD534961DD01E0F704E6A4356DFEA585815` |
| `eval-v4.1-rich-communication.jsonl` | `071E1DA34722226FF1D1EC9947F5DC1881A99061A107DBD2F8619C9C35C549E9` |
| `eval-v4.1-reachability.jsonl` | `6BF5EE52BF76DFE413A0F1F4910ABAD07D4F8B384432F59EF3CF88ACB34B93B6` |
| `adapter_model.safetensors` | `40353DF97BD33FCCD894F18EE32DA6973DCD14C7D357577DA13EBB0D56666A05` |

Base-model revision:
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

Exact training command:

```powershell
$env:PYTHONPATH = "src"
$env:HF_DATASETS_CACHE = (Resolve-Path "runs\v4.1").Path + "\hf-datasets-cache"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python -m pondllm train `
  --dataset data/generated/sft-v4.1-rich-reachability.jsonl `
  --output artifacts/qwen3.5-0.8b-base-action-sft-v4.1-rich-reachability `
  --model Qwen/Qwen3.5-0.8B-Base `
  --seed 7 `
  --epochs 2 `
  --micro-batch-size 16 `
  --gradient-accumulation-steps 2
```

Primary output paths:

```text
artifacts/qwen3.5-0.8b-base-action-sft-v4.1-rich-reachability/
runs/v4.1/eval-rich/
runs/v4.1/eval-reachability/
runs/v4.1/eval-v4-communication/
runs/v4.1/eval-v4-survival/
runs/v4.1/eval-legacy/
runs/v4.1/paired-v41/
runs/v4.1/baselines/
runs/v4.1/ordinary/
```

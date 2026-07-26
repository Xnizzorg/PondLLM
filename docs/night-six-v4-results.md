# Night-six results: simulator-native V4

Date: 2026-07-26
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB
Base model: `Qwen/Qwen3.5-0.8B-Base`
Method: fresh QLoRA SFT followed by greedy held-out and paired-world evaluation

## Question

Night five showed that V3 could use a coordinate signal in matched worlds, but stopped signalling
when role-shaped IDs and memories were removed. Free-world probes also exposed resource-irrational
sharing. V4 asks:

> Can a fresh 0.8B adapter learn useful signalling from local information geometry, follow the
> delivered information to food, and avoid sharing that threatens survival without relying on
> task-role shortcuts?

This remains supervised learning on a frozen shared language model. It is not population-scale RL
or evidence of emergent language.

## Curriculum correction

V4 is generated from real `World.observe` snapshots and world transitions. Organism IDs, lineage
IDs, spawn order, tick, age, and plausible lifetime memory are randomized independently of the
expected action. The communication cases include:

- useful sender signals when a nearby recipient cannot see the food;
- redundant-signal controls when the recipient can already see the food;
- informed recipients that move toward a delivered coordinate;
- matched recipients without the delivered message;
- recipients that prefer currently visible food over a stale or incorrect memory;
- complete delivered-message trajectories through movement to a successful forage.

The survival cases pair safe sharing with unsafe-reserve, unnecessary-sharing, and low-energy
food-priority counterfactuals. The legacy V2 corpus is curated rather than copied wholesale:
generic signalling and unsafe or unnecessary sharing examples are removed, then every retained
record is rebuilt with the current prompt.

The final training set contains 27,014 records:

| Component | Records |
|---|---:|
| Curated V2 base | 12,059 |
| Five core communication cases | 5,000 |
| Recipient trajectories | 5,955 |
| Four survival cases | 4,000 |

Corpus audits found zero duplicate records, illegal labels, role-shaped identifiers, age/tick
inconsistencies, impossible memory phrases, or overlap with either held-out V4 set. All 1,000
generated recipient trajectories reached and foraged the intended food in the simulator.

## Predetermined gates

- at least 99% strict JSON and observation-legal model actions;
- at least 70% useful signalling and no more than 10% redundant signalling;
- at least 70% correct signal payload coordinates;
- at least 70% exact recipient behavior in informed, control, and visible-override cases;
- at least 70% safe-share accuracy and no more than 10% sharing in unsafe or unnecessary cases;
- at least 70% low-energy food-priority accuracy;
- no material loss of common legacy actions;
- a substantial normal-versus-blocked forage advantage in neutral paired worlds;
- lower success under corrupted coordinates.

Population-scale RL remains blocked unless the communication and survival gates pass.

## Training

The adapter was trained fresh for two epochs with seed 7, 4-bit NF4 loading, BF16 compute, LoRA
rank 8 and alpha 16 over all linear layers, learning rate `2e-4`, micro-batch size 16, and two
gradient-accumulation steps. The effective batch size was 32. Completion-only loss was used.

Training completed 1,676 optimizer steps in 2 hours 53 minutes. Aggregate training loss was
0.03519. The fixed 200-record training split had 0.02894 loss and 98.36% mean token accuracy after
epoch one, improving to 0.02469 and 98.63% after epoch two.

## Results

### Neutral held-out communication

All five cases contain neutral, role-independent IDs and lineages, mixed plausible memory, and
randomized tick, age, and spawn order. None overlaps the training set.

| Metric | Gate | V4 |
|---|---:|---:|
| Strict JSON | at least 99% | 100% |
| Observation-legal action | at least 99% | 100% |
| Useful signal | at least 70% | 98% |
| Redundant signal | at most 10% | 1% |
| Correct payload coordinate | at least 70% | 98% |
| Recipient informed exact action | at least 70% | 98% |
| Recipient no-message control exact action | at least 70% | 78.5% |
| Visible-food override exact action | at least 70% | 99.5% |

Overall exact action accuracy was 94.6%. The no-message control is the weakest case: 36 expected
moves became rests and five expected rests became moves. This clears the gate but remains the
largest static error mode.

### Survival counterfactuals

| Case | Gate | Exact | Share rate |
|---|---:|---:|---:|
| Safe share | at least 70% exact | 100% | 100% |
| Unsafe reserve | at most 10% share | 99.6% | 0.4% |
| Share not needed | at most 10% share | 99.2% | 0% |
| Low-energy food priority | at least 70% exact | 100% | 0% |

The suite was 99.7% exact overall, with 100% strict JSON and 100% legal actions.

### Legacy retention

V4 preserves or improves every common action relative to the 0.8B V3 adapter:

| Expected action | V3 recall | V4 recall |
|---|---:|---:|
| Forage | 100% | 100% |
| Move | 70% | 72% |
| Reproduce | 92% | 98% |
| Rest | 87% | 94% |
| Share | 49% | 0% |
| Generic signal | 0% | 0% |

All 600 V4 outputs were strict JSON and legal. The aggregate legacy score falls from 66.33% to
60.67% because V4 rejects all old share and generic-signal labels. Those labels are intentionally
in conflict with V4: old sharing did not require a needy recipient or a safe donor reserve, and
old signals did not carry useful coordinates. The new controlled survival suite demonstrates that
the adapter can share when the V4 conditions actually hold. The old aggregate should therefore not
be interpreted as an unconditional regression, but it does show that the policy has not learned a
smooth boundary spanning both distributions.

### Neutral paired worlds

Sixteen layouts were run with normal, blocked, corrupted, and one-energy-cost channels. The same
scene state is reused across each four-way intervention.

| Condition | Sender signalled | Recipient foraged | Strict/legal model decisions |
|---|---:|---:|---:|
| Normal | 93.75% | 93.75% | 100% / 100% |
| Blocked | 93.75% | 0% | 100% / 100% |
| Corrupted | 93.75% | 6.25% | 100% / 100% |
| Costly | 93.75% | 93.75% | 100% / 100% |

The normal-minus-blocked forage effect is +93.75 percentage points. Normal recipients finish
3.938 Manhattan cells closer and with 2.375 more energy than blocked recipients on average.
Corruption reduces acquisition to one of 16 worlds. All 15 costly signals were charged.

On the identical seeds, the unadapted 0.8B base had 0% observation-valid model actions, emitted no
signals, acquired no food, and had zero paired effect. The heuristic reference emitted no signals
and had no positive normal-versus-blocked effect. A scripted positive control reaches food in every
normal V4 test scene and none of the blocked scenes, establishing that the harness itself is
sensitive to the intended mechanism.

This is the principal V4 success: the neutral result is not tied to role-shaped identifiers,
blocking delivery removes the benefit, and corrupting coordinates nearly removes it.

### Ordinary fixed-seed worlds

The adapter was also run for 30 ticks on seeds 7–10.

| Seed | Decisions | Births | Deaths | Living | Shares | Signals |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 143 | 3 | 0 | 7 | 0 | 0 |
| 8 | 138 | 2 | 0 | 6 | 0 | 0 |
| 9 | 124 | 1 | 1 | 4 | 0 | 0 |
| 10 | 146 | 3 | 1 | 6 | 0 | 0 |
| Total | 551 | 9 | 2 | 23 | 0 | 0 |

Every action was legal. No reciprocal-sharing loop occurred. There were also no observations that
met the strict V4 safe-share condition, so zero shares is appropriate but does not establish live
helping behavior.

Both deaths were four-tick-old offspring. Each began with four energy, moved toward visible food
for four ticks, and died before reaching it. Rest would have offset metabolism, while movement did
not account for whether the food was reachable within the remaining energy.

More importantly, there were 13 same-tick situations in which an organism saw food while a visible
peer's own observation contained no current or visible food. V4 moved in all 13 and signalled in
none. The controlled two-agent protocol therefore has not yet generalized to richer
multi-agent/multi-food population context.

For context, the existing same-seed heuristic baseline produced 11 births and seven deaths; the
unadapted base produced no births and all 16 founders died. V4 improves this short survival probe,
but four worlds are not a fitness estimate.

## Interpretation and next step

V4 passes every predetermined neutral communication, strict-action, controlled survival, common
legacy-action, and paired causal gate. It repairs V3's role-cue shortcut and removes the observed
resource-irrational sharing loops.

It is not ready for population-scale RL. The next experiment should be a V4.1 SFT correction:

1. sample ordinary multi-agent worlds and build communication counterfactuals from the sender and
   each recipient's contemporaneous real observations;
2. mix multiple agents, multiple food patches, distractor food, current-food cases, and explicit
   action-priority negatives into both training and held-out suites;
3. add reachability-aware offspring trajectories: move only when food can be reached before energy
   exhaustion, otherwise rest or receive a safe parental share;
4. create live worlds that deliberately generate safe-share opportunities and measure assistance,
   donor reserve, reciprocal transfers, and offspring survival;
5. require spontaneous signalling in those rich worlds, not only success in the paired harness.

Population-scale RL should remain deferred until that richer live communication and assistance
gate passes. The current V4 adapter is a strong supervised control for that experiment.

## Reproducibility

| Artifact | SHA-256 |
|---|---|
| `sft-v4-simulator-native.jsonl` | `9BDE22C42DC62B66B26449857C8005E41CE5ABEBF9BEA6C16ED72467CBAEF7EE` |
| `eval-v4-communication.jsonl` | `F81B23D44DF6E2560226591492B8C7753F30D9F13BD359C33ED4F6768D08941F` |
| `eval-v4-survival.jsonl` | `74E3916D55493C4C86B3D583CC0A1A85B91955F8BBB37E94E9AEDD2DD96A1D0F` |
| `adapter_model.safetensors` | `E0B76547781DB57F94D19175A9791430362BC9B5355EA7B221F67B5C4477E82B` |

Base-model revision:
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Training ran from Git commit
`c5eeca909775acfa99342795566323de9ddb76bd` with the documented V4 worktree changes uncommitted;
the run manifest records that dirty state.

Exact training command:

```powershell
python -m pondllm train `
  --dataset data/generated/sft-v4-simulator-native.jsonl `
  --output artifacts/qwen3.5-0.8b-base-action-sft-v4-simulator-native `
  --model Qwen/Qwen3.5-0.8B-Base `
  --seed 7 `
  --epochs 2 `
  --micro-batch-size 16 `
  --gradient-accumulation-steps 2
```

Critical paired-world gate:

```powershell
python -m pondllm run-communication-world `
  --profile v4 `
  --scenes 16 `
  --seed-start 300 `
  --steps 11 `
  --temperature 0 `
  --model Qwen/Qwen3.5-0.8B-Base `
  --adapter artifacts/qwen3.5-0.8b-base-action-sft-v4-simulator-native `
  --output runs/communication-world-0.8b-v4-neutral
```

Primary output paths:

```text
artifacts/qwen3.5-0.8b-base-action-sft-v4-simulator-native/
runs/qwen35-0.8b-v4-communication/
runs/qwen35-0.8b-v4-survival/
runs/qwen35-0.8b-v4-legacy-action/
runs/communication-world-0.8b-v4-neutral/
runs/communication-world-0.8b-unadapted-v4-neutral/
runs/communication-world-heuristic-v4-neutral/
runs/qwen35-0.8b-v4-live-seed-{7,8,9,10}/
```

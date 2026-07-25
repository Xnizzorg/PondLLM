# Night-five results: world-level communication ablation

Date: 2026-07-25  
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB  
Models: the existing Qwen3.5 0.8B and 2B V3 communication adapters  
Method: greedy decoding in paired deterministic worlds

## Question

Night four showed high accuracy on disjoint examples from the V3 communication generator. This
experiment asks a stronger question:

> Does a delivered model-generated signal change another model-controlled organism's behavior and
> ecological outcome, and does that effect survive removal of training-shaped context cues?

This remains a supervised protocol. It is not emergent language or population-scale evolution.

## Paired world

Four fixed-seed scenes rotate the same information geometry:

- the sender sees both a food patch and a nearby recipient;
- the recipient sees the sender but cannot see the food;
- the sender receives one model-controlled action at the initial tick, then rests;
- the recipient remains model-controlled for seven ticks;
- normal, blocked, corrupted, and one-energy-cost signal channels begin from identical state.

Holding the sender stationary after its first decision prevents it from consuming or physically
blocking the target food. The blocked condition distinguishes information transfer from a
recipient finding food by chance. Corruption deterministically shifts a `food at [x,y]` payload
to a distant coordinate. The costly channel delivers the correct payload and charges the sender
one energy.

A scripted sender/follower policy is part of the test suite. It reaches food in every normal
scene, never does so when delivery is blocked, performs worse under corruption, and pays the
configured signal cost. This establishes that the intervention itself can produce and measure
the intended causal effect.

Every decision, message, delivered or modified payload, energy cost, movement, forage, and final
state is written to JSONL/JSON.

## Two context profiles

The `matched` profile deliberately preserves context patterns from the V3 generator: sender and
recipient prefixes in organism and lineage IDs, a nonzero tick, and role-correlated memory phrases
such as `last forage was ...` and `survived tick ...`.

The `clean` profile keeps the same geometry and energies but uses neutral sequential organism and
lineage IDs, tick zero, and empty memories. Only the observation's actual information content can
justify signalling.

This is a schema-generalization test, not merely a new random seed.

## Results

Each row contains four normal scenes and its paired blocked, corrupted, and costly copies.

| Adapter/profile | Valid | Sender signalled | Normal foraged | Blocked foraged | Corrupted foraged | Costly foraged | Normal - blocked forage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B matched | 100% | 75% | 75% | 0% | 50% | 75% | +75 pp |
| 2B matched | 100% | 100% | 100% | 0% | 50% | 100% | +100 pp |
| 0.8B clean | 100% | 0% | 25% | 25% | 25% | 25% | 0 pp |
| 2B clean | 100% | 0% | 0% | 0% | 0% | 0% | 0 pp |

Matched-profile paired effects:

| Adapter | Normal distance advantage | Normal recipient-energy advantage | Costly signals charged |
|---|---:|---:|---:|
| 0.8B | 3.00 cells | +2.25 energy | 3/4 |
| 2B | 4.00 cells | +2.75 energy | 4/4 |

The 0.8B clean recipient happened to find food in one scene, but it did so identically with normal,
blocked, corrupted, and costly channels. The paired effect is therefore zero rather than evidence
of communication.

## Interpretation

The matched result is real within its scope. When the adapters emit the trained payload, delivery
causes the recipient to find and forage food. Blocking removes that benefit, corruption weakens
it, and charging the sender reduces its energy without removing the recipient benefit. The 2B
adapter executes this trained protocol more consistently than the 0.8B adapter.

The clean result fails completely. Neither adapter signals once the generator's role-shaped IDs,
lineages, tick, and memory phrases are removed. Increasing model size from 0.8B to 2B did not fix
the failure.

The night-four static scores remain valid measurements of in-generator accuracy, but they
overstated live robustness. The held-out set used disjoint seeds while retaining the same
generator schema, including `organism-s...`, `organism-r...`, `lineage-s...`, `lineage-r...`, and
role-correlated memory. The adapters learned both a useful protocol and a shortcut for recognizing
when to invoke it.

Four scenes are enough to expose this binary failure but not enough to estimate population fitness
or subtle effect sizes. The V3 adapters therefore remain useful controls, not finished ecological
policies.

## Free-world probe

Before the paired harness was added, the adapters were also run in ordinary 20-tick worlds:

| Adapter | Worlds | Decisions | Valid | Signals | Births | Deaths |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B V3 | 3 | 263 | 263 | 0 | 5 | 2 |
| 2B V3 | 1 | 94 | 94 | 0 | 3 | 2 |

Two 0.8B worlds entered repeated reciprocal-sharing loops and a donor died. In the 2B world a
founder repeatedly shared, then died; a young organism also failed to recover from low energy.
These are exploratory observations rather than a controlled survival score, but they identify a
second curriculum weakness: legal sharing is not yet resource-rational sharing.

## V4 curriculum

V4 should be trained fresh from a corrected corpus rather than patched with simulator guards:

1. Randomize organism and lineage IDs independently of behavioral role.
2. Mix empty, irrelevant, and ordinary memories across sender and recipient cases.
3. Randomize ticks and ages so neither predicts the action class.
4. Preserve useful/redundant counterfactual pairs, but require geometry and perception radius to
   determine whether the recipient lacks information.
5. Include complete `signal -> delivered memory -> recipient movement -> forage` trajectories with
   multiple layouts, agents, and food patches.
6. Add paired survival examples: do not share below an energy reserve, do not repeat transfers
   that endanger the donor, forage or rest after reproduction when depleted, and prioritize
   recovery for low-energy offspring.

The next gate should require:

- at least 99% syntax and observation legality;
- at least 70% useful signalling with neutral IDs and mixed memory;
- no more than 10% redundant signalling;
- at least 70% correct payload coordinates;
- at least 70% recipient food acquisition in the clean normal channel;
- a substantial normal-versus-blocked paired advantage;
- lower success under corrupted coordinates;
- no loss of common legacy actions;
- a fixed survival suite with low-energy and repeated-sharing counterfactuals.

Do not begin population-scale RL until a V4 SFT adapter passes both the clean communication and
survival gates.

## Reproduction

Run one profile with:

```powershell
python -m pondllm run-communication-world `
  --model Qwen/Qwen3.5-0.8B-Base `
  --adapter artifacts/qwen3.5-0.8b-base-action-sft-v3-communication `
  --steps 7 `
  --scenes 4 `
  --seed-start 100 `
  --profile clean `
  --temperature 0 `
  --output runs/communication-world-0.8b-clean
```

Primary result directories:

```text
runs/communication-world-0.8b-matched-v4/
runs/communication-world-0.8b-clean-final/
runs/communication-world-2b-matched-v4/
runs/communication-world-2b-clean-final/
```

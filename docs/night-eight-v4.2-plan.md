# Night eight: V4.2 frozen plan and gates

V4.2 is a fresh Qwen3.5 0.8B Base SFT adapter. It is not initialized from the V4.1 adapter.
The correction targets the two V4.1 failures: 18% redundant signalling in rich contexts and
energy-boundary moves that caused three ordinary-world deaths.

## Corpus

The training corpus keeps the audited V4.1 records and normalizes them to the V4.2 system prompt.
It adds:

- the 36 V4.1 redundant-signal errors as mined hard negatives; these records are excluded from
  all V4.2 primary held-out suites;
- 1,000 disjoint rich useful/redundant pairs;
- 3,000 additional rich redundant cases, making the correction deliberately negative-heavy;
- 1,000 simulator-native energy sequences covering exact-boundary waiting, one-point-above
  movement, eventual forage, distractors, prior signals, prior shares, young and older actors,
  safe rescue, no repeated transfer, unsafe-donor control, and unrescued waiting.

The actual default experiment has `move_cost=0`, `metabolism=1`, and `rest_gain=1`. Food at
Manhattan distance `d` therefore requires `energy > d`: after `d` move turns, the actor needs to
remain alive for the later forage turn. This formula was rechecked against the simulator before
V4.2 labels were generated.

Primary V4.2 held-outs use new seeds and have zero semantic prompt/completion overlap with
training. The old V4.1 rich suite is training-contaminated by intentional error mining and is
reported only as a diagnostic, never as a primary gate.

## Frozen gates

These thresholds are fixed before training or model evaluation:

1. At least 99% strict JSON and 99% observation-legal actions in every primary static and live
   suite.
2. In the new rich held-out suite: at least 75% useful signalling, at most 10% redundant
   signalling, and at least 75% correct signal coordinates.
3. At least 90% exact behavior in every V4.2 energy and rescue case.
4. Retain V4.1's original controlled behavior: at least 95% exact on V4 communication and V4
   survival, at least 95% forage and reproduce recall, at least 65% move recall, and at least 90%
   rest recall on the legacy suite.
5. In 16 new rich paired worlds: at least a +50-point normal-minus-blocked forage effect, with
   lower success under corrupted coordinates.
6. In 16 live rescue scenes:
   - safe donors share and rescued children forage in at least 75% of normal scenes;
   - the normal-minus-blocked child-forage effect is at least +50 points;
   - blocked children wait and survive in at least 90% of scenes;
   - there are zero energy-boundary child moves, repeated transfers, and unsafe-donor shares;
   - every donor survives.
7. Across ordinary 30-tick worlds on fixed seeds 7 through 38: zero moves at or below the
   move-then-forage energy boundary, fewer deaths than V4.1 on the same seeds, and at least one
   signal in a strict information-asymmetry opportunity.
8. Population-scale RL remains blocked if any controlled or ordinary-world gate fails.

## Training recipe

- model: `Qwen/Qwen3.5-0.8B-Base`
- base-model revision: resolved and recorded by the training manifest
- seed: 7
- epochs: 2
- completion-only loss
- 4-bit NF4 base loading with BF16 compute
- LoRA rank 8, alpha 16, dropout 0.05 over all linear layers
- learning rate: `2e-4`
- micro-batch size: 16
- gradient accumulation: 2

The corpus audit, git revision, dirty state, dataset and adapter SHA-256 hashes, package versions,
elapsed time, and all pass/fail results will be recorded after the run.

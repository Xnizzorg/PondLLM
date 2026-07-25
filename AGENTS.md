# PondLLM engineering notes

PondLLM explores artificial ecology using a frozen shared language model and small inherited
LoRA adapters. Preserve these boundaries:

- The simulator is deterministic under a fixed seed and has no ML dependencies.
- An organism can only act from its local `Observation`.
- Lifetime memory belongs to an organism. Adapter state belongs to a lineage.
- Evolutionary updates happen between evaluation windows, never after every action.
- Every model action, parse failure, world transition, birth, and death must be loggable.
- New training methods must be evaluated against fixed-seed heuristic and unadapted baselines.

Run simulator tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Do not add population-scale RL until one SFT adapter passes the action-validity evaluation.


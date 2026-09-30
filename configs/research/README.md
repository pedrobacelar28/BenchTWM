# Research Recipes

Research definitions are small JSON matrices over catalog dataset IDs, public
method IDs, and explicit seeds. Run one with:

```bash
python scripts/bench.py research matrix configs/research/v0-offline-matrix.json --execute
python scripts/bench.py research summarize
```

Use `research sweep` for one method across seeds and `research tune` for the
Optuna search space declared by the CLI for that algorithm family. The v0
matrix has exactly two tasks, four direct baselines, three provisional
planning methods per environment, and seeds `0,1,2`.

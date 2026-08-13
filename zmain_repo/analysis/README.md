# Analysis

BenchTWM exports machine-readable training histories, direct simulator episodes,
trajectories, OPE summaries, and run manifests under `artifacts/`. Generate a
compact environment-agnostic report with:

```bash
python scripts/bench.py report <run-id-or-path>
```

Use `python scripts/bench.py research collect` to create a flat CSV index over
all clean-layout runs. Downstream notebooks and plotting code should consume
the exported CSV/JSON artifacts rather than depend on private runner objects.

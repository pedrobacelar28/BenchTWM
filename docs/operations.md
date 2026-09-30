# Operations

Run all commands from the BenchTWM repository root with:

```bash
python scripts/bench.py <command>
```

`doctor` is the first diagnostic command. It reports Python packages, CUDA,
visible GPUs, the artifact root, and the vendored ICU-Sepsis and EpiCare cores.

## Offline Lifecycle

```bash
python scripts/bench.py catalog
python scripts/bench.py dataset show epicare/smart/dev
python scripts/bench.py dataset build epicare/smart/dev
python scripts/bench.py train epicare/smart/dev dqn --dry-run
python scripts/bench.py train epicare/smart/dev dqn
python scripts/bench.py status
```

`train` checks that `train.npz`, `val.npz`, and, when requested, `ope_test.npz`
exist before it starts. Dataset generation is always explicit.

For offline recipes, the initial plan states the early-stopping monitor,
direction, validation cadence, patience, and checkpoint rule. The terminal then
labels optimization, final direct evaluation, and held-out OPE as distinct
phases. OPE shows separate ratio-construction and bootstrap progress bars.

Use `--set KEY=JSON_VALUE` for an ephemeral typed override. It never mutates a
recipe file:

```bash
python scripts/bench.py train epicare/smart/dev dqn \
  --set offline_train_steps=2000 --set policy_hidden_dim=64 --dry-run
```

Use `--mode online` only for deliberate simulator baselines. Offline is the
default and is the recommended benchmark workflow.

## Suites and Local Queues

`suite` only prints its plan unless `--execute` is present:

```bash
python scripts/bench.py suite epicare/smart/dev --method dqn --method raw-model
python scripts/bench.py suite epicare/smart/dev --seeds 0,1 --jobs 2 --execute
python scripts/bench.py suite epicare/smart/dev --seeds 0,1 --jobs 2 --gpus 0,1 --execute
python scripts/bench.py suite epicare/smart/dev --seeds 0,1 --jobs 2 --gpus 0,1 --follow --execute
```

Each worker receives an isolated resolved config, log file, manifest, and
device assignment. A completed run is skipped on a later queue invocation
unless `--force` is supplied. With `--follow`, the parent process renders one
live progress row per job without interleaving worker output. Queue manifests
and the central `suite_results.json` are under `artifacts/queues/<timestamp>/`.
The central JSON contains every job's identity, resolved config, optimization
summary, direct evaluation, OPE, timing, and method-level aggregates. The
individual run directories remain the canonical location for checkpoints and
per-method histories.

## Evidence and Artifacts

The command line deliberately labels three different measurement regimes:

1. **Offline optimization** uses fixed `train.npz` and `val.npz` and reports
   algorithm-specific losses.
2. **Direct simulator evaluation** uses fresh benchmark episodes and reports
   return, success, safety, burden, and environment-declared metrics.
3. **OPE** uses held-out `ope_test.npz`; IS, WIS, PDIS, WPDIS, bootstrap
   intervals, support, ESS, and extreme-weight diagnostics are reported
   independently from optimization and simulator evaluation.

```bash
python scripts/bench.py inspect run <run-id-or-path>
python scripts/bench.py evaluate <run-id-or-path> --episodes 1000
python scripts/bench.py ope <run-id-or-path> --bootstrap 500
python scripts/bench.py report <run-id-or-path>
```

`ope` uses the deployed policy semantics by default. Its reliability table must
be checked before treating an estimate as a ranking. `--target-mode stochastic`
evaluates a different counterfactual stochastic policy, while
`--write-canonical` replaces the run-level OPE summary after a deployed-mode
recomputation.

`status`, `catalog`, `inspect`, and the final results of operational commands
accept `--json` for automation.

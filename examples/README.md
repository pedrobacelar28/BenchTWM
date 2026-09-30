# Examples

All examples run from the repository root and use only the public
`python scripts/bench.py` interface.

| File | Demonstrates |
|---|---|
| `01_epicare_offline.sh` | Explicit logged-data build and offline training. |
| `02_inspect_dataset.py` | Reading the versioned NPZ schema and metadata. |
| `03_evaluate_and_ope.sh` | Fresh simulator evaluation, OPE, and a report for a run. |
| `04_local_suite.sh` | Dry-running and executing a local multi-seed queue. |
| `05_extension_check.py` | Validating the catalog and adapter surface before an extension. |

The shell examples intentionally use EpiCare `dev` and dry-run commands first.
Replace the IDs with any values shown by `python scripts/bench.py catalog`.

# Research Matrices

A matrix is a JSON file with `datasets`, optional `methods` (default: every
compatible method), and `seeds`; or a list of such blocks under `jobs`.

```bash
bench research matrix configs/research/dev-matrix.json             # print the plan
bench research matrix configs/research/dev-matrix.json --execute   # run it
bench research summarize
```

- `dev-matrix.json`: every method on the main behavior dataset of each
  environment (`epicare/smart/dev`, `icu-sepsis/expert/dev`), seeds 0-2.
- `final-matrix.json`: every method on every `final` dataset, seeds 0-2.

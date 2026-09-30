# Documentation Map

The documentation is organized around the lifecycle of a benchmark experiment.

1. [Operations](operations.md): the offline-first `bench.py` lifecycle,
   queues, output inspection, and evidence boundaries.
2. [Benchmarks](benchmarks.md): ICU-Sepsis, EpiCare, and the two DTR additions,
   their source material, observation/action spaces, and the adapter contract.
3. [Datasets](datasets.md): curated behavior-policy variants, `dev`/`final`
   tiers, and public dataset IDs.
4. [Data schema](data-schema.md): upstream data formats, normalized local NPZ
   files, behavior propensities, splits, and metadata.
5. [Algorithms](algorithms.md): DQN, PPO, the three CEM world-model families,
   and the offline DreamerV3-style agent.
6. [Dreamer fidelity audit](dreamer-fidelity.md): source commit, module mapping,
   BenchTWM boundary adaptations, and parity requirements.
7. [Evaluation](evaluation.md): training metrics, simulator evaluation, OPE,
   artifacts, and interpretation limits.
8. [Extending the suite](extending.md): the exact contracts for a new
   environment or algorithm.
9. [Recipes and research](recipes.md): catalog IDs, versioned recipes, sweeps,
   matrices, and tuning.
10. [Technical specification](paper/main.tex): a self-contained LaTex document
   written in paper form, without empirical results.

The descriptions distinguish three things throughout:

- **Source benchmark**: what the upstream repository and its paper define.
- **Suite integration**: the API, data format, and configuration used here.
- **Local algorithms**: methods implemented in this repository, which are not
  necessarily the same algorithms reported in the source papers.

This distinction matters particularly for EpiCare: its source paper reports
AWAC, EDAC, TD3+BC, IQL, CQL, BC, and DQN, whereas this suite currently runs
DQN, PPO, three world-model/CEM variants, and the offline DreamerV3-style agent
on the EpiCare environment.

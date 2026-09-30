# Documentation

Read in the order of an experiment's lifecycle:

1. [Environments](environments.md): ICU-Sepsis and EpiCare, their observations,
   actions, metrics, and the adapter contract.
2. [Datasets](datasets.md): behavior policies, tiers, and dataset ids.
3. [Data schema](data-schema.md): the arrays stored in every split.
4. [Algorithms](algorithms.md): the eight methods and the shared interface.
5. [Evaluation](evaluation.md): model selection, direct simulator evaluation,
   OPE, and the planning-gain summary.
6. [Configuration](configuration.md): how a run config is resolved from layers.
7. [Command line and artifacts](operations.md): every `bench` command and the
   files a run writes.
8. [Extending](extending.md): adding an environment or an algorithm.
9. [DreamerV3 fidelity audit](dreamer-fidelity.md).
10. [Technical specification](paper/main.tex): the benchmark in paper form,
    without empirical results.
11. [Research notes](research/): candidate structured environments and
    datasets ([CATALOGO_AMBIENTES.md](research/CATALOGO_AMBIENTES.md),
    Portuguese) and the text of the ICU-Sepsis and EpiCare papers (`papers/`).

Throughout, three things are kept apart:

- **Source benchmark**: what the upstream environment and its paper define.
- **Integration**: the adapter, dataset schema, and configuration used here.
- **Local algorithms**: methods implemented in this repository, which are not
  necessarily the ones reported in the source papers. EpiCare's paper, for
  example, reports AWAC, EDAC, TD3+BC, IQL, CQL, BC, and DQN; BenchTWM runs its
  own BC, DQN, IQL, CQL, three CEM world models, and a DreamerV3-style agent.

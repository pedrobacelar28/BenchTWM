"""Check the registries and configs after adding an environment or algorithm."""

from benchtwm.algorithms import algorithms
from benchtwm.config import Benchmark
from benchtwm.envs import environment_ids

bench = Benchmark.load()
errors = bench.validate()
if errors:
    raise SystemExit("\n".join(errors))

print("environments:", ", ".join(environment_ids()))
print("algorithms:", ", ".join(f"{cls.method_id} ({cls.family})" for cls in algorithms()))
for dataset in bench.datasets:
    print(dataset.identifier, "->", ", ".join(method.method_id for method in bench.methods(dataset.identifier)))
print("Configuration is valid. Next: bench validate --smoke && pytest")

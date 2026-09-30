"""Minimal preflight used after registering a new environment or algorithm."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchtwm.operations.catalog import Catalog


catalog = Catalog.load()
errors = catalog.validate()
if errors:
    raise SystemExit("\n".join(errors))

for dataset in catalog.datasets:
    methods = catalog.compatible_methods(dataset.identifier, mode="offline")
    print(dataset.identifier, "->", ", ".join(method.method_id for method in methods))

print("Catalog is valid. Run: python scripts/bench.py validate --smoke")

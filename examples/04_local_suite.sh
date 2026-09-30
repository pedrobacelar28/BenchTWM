#!/usr/bin/env bash
set -euo pipefail

python scripts/bench.py suite epicare/smart/dev --method dqn --method raw-model --seeds 0,1 --dry-run
python scripts/bench.py suite epicare/smart/dev --method dqn --method raw-model --seeds 0,1 --jobs 2 --execute

# On a host with two visible GPUs, use:
# python scripts/bench.py suite epicare/smart/dev --method dqn --seeds 0,1 --jobs 2 --gpus 0,1 --execute

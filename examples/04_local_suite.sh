#!/usr/bin/env bash
# Plan, then run, a two-method, two-seed queue on one dataset.
set -euo pipefail

bench suite epicare/smart/dev --method dqn --method raw-model --seeds 0,1
bench suite epicare/smart/dev --method dqn --method raw-model --seeds 0,1 --jobs 2 --execute
bench research summarize --dataset epicare/smart/dev

# With two visible GPUs:
# bench suite epicare/smart/dev --seeds 0,1,2 --jobs 2 --gpus 0,1 --follow --execute

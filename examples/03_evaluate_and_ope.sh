#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-id-or-path>" >&2
  exit 2
fi

run="$1"
python scripts/bench.py inspect run "$run"
python scripts/bench.py evaluate "$run" --episodes 500
python scripts/bench.py ope "$run" --bootstrap 500
python scripts/bench.py report "$run"

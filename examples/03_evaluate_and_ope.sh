#!/usr/bin/env bash
# Re-evaluate a completed run in the simulator, recompute OPE, and write a report.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-id-or-path>" >&2
  exit 2
fi

run="$1"
bench inspect run "$run"
bench evaluate "$run" --episodes 500
bench ope "$run" --bootstrap 500
bench report "$run"

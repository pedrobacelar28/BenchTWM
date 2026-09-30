#!/usr/bin/env bash
set -euo pipefail

python scripts/bench.py doctor
python scripts/bench.py dataset build epicare/smart/dev --dry-run
python scripts/bench.py dataset build epicare/smart/dev
python scripts/bench.py train epicare/smart/dev dqn --dry-run
python scripts/bench.py train epicare/smart/dev dqn
python scripts/bench.py status

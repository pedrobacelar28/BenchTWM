#!/usr/bin/env bash
# Build a small dataset, inspect a training plan, then train, evaluate, and run OPE.
set -euo pipefail

bench doctor
bench dataset build epicare/smart/dev --dry-run
bench dataset build epicare/smart/dev
bench train epicare/smart/dev dqn --dry-run
bench train epicare/smart/dev dqn
bench status

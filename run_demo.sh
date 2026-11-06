#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the airflow demo.
# Usage: ./run_demo.sh path/to/mesh.stl [--res 100] [--U 1.0]

MESH="$1"
shift || true

if [ -z "${MESH-}" ]; then
  echo "Usage: $0 path/to/mesh.stl [--res N] [--U U]"
  exit 1
fi

# Create and activate virtualenv if needed
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate

# Ensure dependencies installed (idempotent)
pip install -r requirements.txt

# Run the simulator with remaining arguments
python simulate_flow.py --mesh "$MESH" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHYSICELL_DIR="${PHYSICELL_DIR:-$ROOT/external/PhysiCell}"
MANIFEST="${1:?usage: run_instance.sh MANIFEST [zero_based_index]}"
TASK_ID="${2:-${TASK_INDEX:-0}}"

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

INST="$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")"

if [ -z "$INST" ]; then
  echo "No instance found for index $TASK_ID in $MANIFEST" >&2
  exit 2
fi

if [ ! -x "$PHYSICELL_DIR/cancer_biorobots" ]; then
  echo "PhysiCell executable not found: $PHYSICELL_DIR/cancer_biorobots" >&2
  echo "Run scripts/physicell/setup_physicell_project.sh first, or set PHYSICELL_DIR." >&2
  exit 2
fi

if [ ! -f "$INST/config.xml" ]; then
  echo "Instance config not found: $INST/config.xml" >&2
  exit 2
fi

cd "$PHYSICELL_DIR"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" ./cancer_biorobots "$INST/config.xml" > "$INST/run.log" 2>&1
echo "completed $INST"

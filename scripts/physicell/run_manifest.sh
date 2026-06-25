#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${1:?usage: run_manifest.sh MANIFEST [num_instances]}"
JOBS="${JOBS:-1}"

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

N="${2:-$(wc -l < "$MANIFEST")}"

if ! command -v xargs >/dev/null 2>&1; then
  echo "Required command not found: xargs" >&2
  exit 127
fi

echo "Running $N instances from $MANIFEST with $JOBS worker(s)."
seq 0 "$((N - 1))" | xargs -P "$JOBS" -I {} \
  bash "$ROOT/scripts/physicell/run_instance.sh" "$MANIFEST" {}

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${1:?usage: gen_advantage_shards.sh DATA_ROOT [split] [out_dir] [num_instances] [shard_size]}"
SPLIT="${2:-train}"
OUT_DIR="${3:-$REPO_ROOT/shards}"
N="${4:-1500}"
SHARD_SIZE="${5:-25}"
JOBS="${JOBS:-1}"
PYTHON="${PYTHON:-python}"

if [ ! -f "$DATA_ROOT/$SPLIT/cache.npz" ]; then
  echo "Cache not found: $DATA_ROOT/$SPLIT/cache.npz" >&2
  echo "Run scripts/physicell/cache_fields.py before generating advantage-label shards." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
NSHARDS=$(( (N + SHARD_SIZE - 1) / SHARD_SIZE ))
export REPO_ROOT DATA_ROOT SPLIT OUT_DIR SHARD_SIZE PYTHON

echo "Generating $NSHARDS advantage-label shard(s) from $SPLIT with $JOBS worker(s)."
seq 0 "$((NSHARDS - 1))" | xargs -P "$JOBS" -I {} bash -c '
  set -euo pipefail
  t="$1"
  start=$((t * SHARD_SIZE))
  end=$((start + SHARD_SIZE))
  "$PYTHON" "$REPO_ROOT/scripts/training/gen_advantage_data.py" \
    --root "$DATA_ROOT" --split "$SPLIT" \
    --start "$start" --end "$end" \
    --K 6 --n_base 2 --every 12 --radius -1 \
    --out "$OUT_DIR/adv_$(printf "%03d" "$t").npz"
' _

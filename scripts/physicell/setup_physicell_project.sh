#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHYSICELL_DIR="${1:-$ROOT/external/PhysiCell}"
PHYSICELL_REPO="${PHYSICELL_REPO:-https://github.com/MathCancer/PhysiCell.git}"
PHYSICELL_REF="${PHYSICELL_REF:-69a23dbe}"

for cmd in git make; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 127
  fi
done

if ! command -v "${PHYSICELL_CPP:-g++}" >/dev/null 2>&1; then
  echo "C++ compiler not found: ${PHYSICELL_CPP:-g++}" >&2
  echo "Install a compiler or set PHYSICELL_CPP to a working compiler command." >&2
  exit 127
fi

mkdir -p "$(dirname "$PHYSICELL_DIR")"
if [ ! -d "$PHYSICELL_DIR/.git" ]; then
  git clone "$PHYSICELL_REPO" "$PHYSICELL_DIR"
fi

cd "$PHYSICELL_DIR"
git checkout "$PHYSICELL_REF"

mkdir -p user_projects
rm -rf user_projects/miso_biomarker_fields
cp -R "$ROOT/physicell/miso_biomarker_fields" user_projects/

if ! make load PROJ=miso_biomarker_fields; then
  cp -R user_projects/miso_biomarker_fields/config .
  cp -R user_projects/miso_biomarker_fields/custom_modules .
  cp user_projects/miso_biomarker_fields/Makefile .
  cp user_projects/miso_biomarker_fields/main.cpp .
fi

make
if [ ! -x "$PHYSICELL_DIR/cancer_biorobots" ]; then
  echo "Build finished but cancer_biorobots was not found." >&2
  exit 1
fi
echo "PhysiCell MISO project is ready at $PHYSICELL_DIR"

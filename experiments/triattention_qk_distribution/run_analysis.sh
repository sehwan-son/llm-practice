#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/triattention_qk_distribution"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-$SCRIPT_DIR/models}"
TENSOR="${TENSOR:-both}"
LAYERS="${LAYERS:-0}"
HEADS="${HEADS:-0}"
PROMPT="${PROMPT:-한국어로 자기소개를 두 문장으로 해줘.}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/manual_run}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/analyze_pre_rope_qk.py" \
  --model "$MODEL" \
  --tensor "$TENSOR" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --plot-summary \
  --export-csv \
  "$@"

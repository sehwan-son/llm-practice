#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/triattention_qk_distribution"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-$SCRIPT_DIR/models}"
DEVICE="${DEVICE:-cuda:1,cuda:0}"
LAYERS="${LAYERS:-0}"
HEADS="${HEADS:-0}"
PROMPT="${PROMPT:-한국어로 자기소개를 두 문장으로 해줘.}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_FIELD="${PROMPT_FIELD:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/manual_run}"
TOP_BANDS="${TOP_BANDS:-2}"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_pre_rope_qk.py"
  --model "$MODEL"
  --device "$DEVICE"
  --layers "$LAYERS"
  --heads "$HEADS"
  --prompt "$PROMPT"
  --output-dir "$OUTPUT_DIR"
  --plot-top-bands "$TOP_BANDS"
)

if [[ -n "$PROMPT_FILE" ]]; then
  cmd+=(--prompt-file "$PROMPT_FILE")
fi
if [[ -n "$PROMPT_FIELD" ]]; then
  cmd+=(--prompt-field "$PROMPT_FIELD")
fi

cmd+=("$@")

exec "${cmd[@]}"

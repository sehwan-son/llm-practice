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
KEY_MAGNITUDE_PLOT_KIND="${KEY_MAGNITUDE_PLOT_KIND:-both}"
KEY_MAGNITUDE_MAX_TOKENS="${KEY_MAGNITUDE_MAX_TOKENS:-0}"
KEY_MAGNITUDE_MAX_CHANNELS="${KEY_MAGNITUDE_MAX_CHANNELS:-0}"
KEY_MAGNITUDE_COLOR_QUANTILE="${KEY_MAGNITUDE_COLOR_QUANTILE:-0.999}"
KEY_MAGNITUDE_3D_MAX_TOKENS="${KEY_MAGNITUDE_3D_MAX_TOKENS:-420}"
KEY_MAGNITUDE_3D_MAX_CHANNELS="${KEY_MAGNITUDE_3D_MAX_CHANNELS:-320}"
KEY_MAGNITUDE_3D_ELEV="${KEY_MAGNITUDE_3D_ELEV:-28}"
KEY_MAGNITUDE_3D_AZIM="${KEY_MAGNITUDE_3D_AZIM:--62}"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_pre_rope_qk.py"
  --model "$MODEL"
  --device "$DEVICE"
  --layers "$LAYERS"
  --heads "$HEADS"
  --prompt "$PROMPT"
  --output-dir "$OUTPUT_DIR"
  --plot-top-bands "$TOP_BANDS"
  --key-magnitude-plot-kind "$KEY_MAGNITUDE_PLOT_KIND"
  --key-magnitude-max-tokens "$KEY_MAGNITUDE_MAX_TOKENS"
  --key-magnitude-max-channels "$KEY_MAGNITUDE_MAX_CHANNELS"
  --key-magnitude-color-quantile "$KEY_MAGNITUDE_COLOR_QUANTILE"
  --key-magnitude-3d-max-tokens "$KEY_MAGNITUDE_3D_MAX_TOKENS"
  --key-magnitude-3d-max-channels "$KEY_MAGNITUDE_3D_MAX_CHANNELS"
  --key-magnitude-3d-elev "$KEY_MAGNITUDE_3D_ELEV"
  --key-magnitude-3d-azim "$KEY_MAGNITUDE_3D_AZIM"
)

if [[ -n "$PROMPT_FILE" ]]; then
  cmd+=(--prompt-file "$PROMPT_FILE")
fi
if [[ -n "$PROMPT_FIELD" ]]; then
  cmd+=(--prompt-field "$PROMPT_FIELD")
fi

cmd+=("$@")

exec "${cmd[@]}"

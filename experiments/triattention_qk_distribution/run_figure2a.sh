#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/triattention_qk_distribution"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-$SCRIPT_DIR/models}"
LAYERS="${LAYERS:-0}"
HEADS="${HEADS:-0}"
PROMPT="${PROMPT:-한국어로 자기소개를 두 문장으로 해줘.}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a concise and helpful assistant.}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/figure2a}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PLOT_MODE="${PLOT_MODE:-figure2a}"
TOP_BANDS="${TOP_BANDS:-1}"
DOMINANT_BAND_METRIC="${DOMINANT_BAND_METRIC:-center_product}"
PLOT_MAX_POINTS="${PLOT_MAX_POINTS:-2000}"
PLOT_RADIUS_QUANTILE="${PLOT_RADIUS_QUANTILE:-0.995}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
NO_CHAT_TEMPLATE="${NO_CHAT_TEMPLATE:-0}"
PLOT_SUMMARY="${PLOT_SUMMARY:-1}"
EXPORT_CSV="${EXPORT_CSV:-1}"
SAVE_COMPLEX_TENSORS="${SAVE_COMPLEX_TENSORS:-0}"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash experiments/triattention_qk_distribution/run_figure2a.sh [options] [-- extra analyzer args]

Draw TriAttention Figure 2(A)-style pre-RoPE Q/K complex-plane plots.

Options:
  --model MODEL                 Hugging Face model id or local model path
  --layers LAYERS               Layer indices, e.g. 0, 0,1,2, or all
  --heads HEADS                 Query head indices, e.g. 0, 0,5,10, or all
  --prompt TEXT                 Prompt text for the calibration forward pass
  --system-prompt TEXT          System prompt used with chat templates
  --output-dir DIR              Directory for generated plots and summaries
  --device DEVICE               auto, cpu, cuda, cuda:0, ...
  --dtype DTYPE                 auto, float16, float32, or bfloat16
  --max-length N                Maximum tokenized prompt length
  --plot-mode MODE              figure2a or both
  --top-bands N                 Number of dominant Q/K RoPE bands to plot
  --dominant-band-metric METRIC center_product or mean_abs_product
  --max-points N                Maximum sampled points per Q or K cloud
  --radius-quantile Q           Axis limit quantile, e.g. 0.995
  --trust-remote-code           Pass trust_remote_code=True to transformers
  --no-chat-template            Use prompt as-is without tokenizer chat template
  --no-summary                  Do not create summary heatmaps/trends
  --no-csv                      Do not export head_metrics.csv/layer_metrics.csv
  --save-complex-tensors        Save complex_pairs.pt
  --python PYTHON               Python executable to use
  -h, --help                    Show this help

Environment variables with the same names as the upper-case settings are also
supported, for example MODEL, LAYERS, HEADS, OUTPUT_DIR, TOP_BANDS,
DOMINANT_BAND_METRIC, PYTHON_BIN.

Examples:
  bash experiments/triattention_qk_distribution/run_figure2a.sh \
    --layers 0 --heads 0 --top-bands 1

  MODEL=experiments/triattention_qk_distribution/models LAYERS=all HEADS=all TOP_BANDS=1 \
    bash experiments/triattention_qk_distribution/run_figure2a.sh
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    printf 'Missing value for %s\n' "$option" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --layers)
      require_value "$1" "${2:-}"
      LAYERS="$2"
      shift 2
      ;;
    --heads)
      require_value "$1" "${2:-}"
      HEADS="$2"
      shift 2
      ;;
    --prompt)
      require_value "$1" "${2:-}"
      PROMPT="$2"
      shift 2
      ;;
    --system-prompt)
      require_value "$1" "${2:-}"
      SYSTEM_PROMPT="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      require_value "$1" "${2:-}"
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      require_value "$1" "${2:-}"
      DTYPE="$2"
      shift 2
      ;;
    --max-length)
      require_value "$1" "${2:-}"
      MAX_LENGTH="$2"
      shift 2
      ;;
    --plot-mode)
      require_value "$1" "${2:-}"
      PLOT_MODE="$2"
      shift 2
      ;;
    --top-bands | --plot-top-bands)
      require_value "$1" "${2:-}"
      TOP_BANDS="$2"
      shift 2
      ;;
    --dominant-band-metric)
      require_value "$1" "${2:-}"
      DOMINANT_BAND_METRIC="$2"
      shift 2
      ;;
    --max-points | --plot-max-points)
      require_value "$1" "${2:-}"
      PLOT_MAX_POINTS="$2"
      shift 2
      ;;
    --radius-quantile | --plot-radius-quantile)
      require_value "$1" "${2:-}"
      PLOT_RADIUS_QUANTILE="$2"
      shift 2
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE=1
      shift
      ;;
    --no-chat-template)
      NO_CHAT_TEMPLATE=1
      shift
      ;;
    --no-summary)
      PLOT_SUMMARY=0
      shift
      ;;
    --no-csv)
      EXPORT_CSV=0
      shift
      ;;
    --save-complex-tensors)
      SAVE_COMPLEX_TENSORS=1
      shift
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_pre_rope_qk.py"
  --model "$MODEL"
  --tensor both
  --layers "$LAYERS"
  --heads "$HEADS"
  --prompt "$PROMPT"
  --system-prompt "$SYSTEM_PROMPT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-length "$MAX_LENGTH"
  --output-dir "$OUTPUT_DIR"
  --plot
  --plot-mode "$PLOT_MODE"
  --plot-top-bands "$TOP_BANDS"
  --dominant-band-metric "$DOMINANT_BAND_METRIC"
  --plot-max-points "$PLOT_MAX_POINTS"
  --plot-radius-quantile "$PLOT_RADIUS_QUANTILE"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ "$NO_CHAT_TEMPLATE" == "1" ]]; then
  cmd+=(--no-chat-template)
fi
if [[ "$PLOT_SUMMARY" == "1" ]]; then
  cmd+=(--plot-summary)
fi
if [[ "$EXPORT_CSV" == "1" ]]; then
  cmd+=(--export-csv)
fi
if [[ "$SAVE_COMPLEX_TENSORS" == "1" ]]; then
  cmd+=(--save-complex-tensors)
fi

cmd+=("${EXTRA_ARGS[@]}")

printf 'Running Figure 2(A)-style Q/K plot\n' >&2
printf '  model: %s\n' "$MODEL" >&2
printf '  layers: %s\n' "$LAYERS" >&2
printf '  heads: %s\n' "$HEADS" >&2
printf '  dominant band metric: %s\n' "$DOMINANT_BAND_METRIC" >&2
printf '  output: %s\n' "$OUTPUT_DIR" >&2

exec "${cmd[@]}"

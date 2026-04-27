#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/triattention_qk_distribution"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage: clean_outputs.sh [--yes] [--output-dir PATH]

Delete generated analysis results under outputs/.

Options:
  --yes              Actually delete files. Without this, only print targets.
  --output-dir PATH  Directory to clean. Defaults to this experiment's outputs/.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      DRY_RUN=0
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

OUTPUT_ROOT="$(realpath -m "$SCRIPT_DIR/outputs")"
TARGET_DIR="$(realpath -m "$OUTPUT_DIR")"

case "$TARGET_DIR" in
  "$OUTPUT_ROOT"|"$OUTPUT_ROOT"/*)
    ;;
  *)
    echo "Refusing to clean outside outputs/: $TARGET_DIR" >&2
    exit 2
    ;;
esac

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Nothing to clean: $TARGET_DIR does not exist."
  exit 0
fi

targets=()
while IFS= read -r -d '' path; do
  targets+=("$path")
done < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 ! -path "$OUTPUT_ROOT/README.md" -print0)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "Nothing to clean under $TARGET_DIR."
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run. These paths would be deleted:"
  printf '  %s\n' "${targets[@]}"
  echo
  echo "Run with --yes to delete them."
  exit 0
fi

rm -rf -- "${targets[@]}"
echo "Deleted ${#targets[@]} path(s) under $TARGET_DIR."

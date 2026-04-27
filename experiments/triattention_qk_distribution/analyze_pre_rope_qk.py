import argparse
from pathlib import Path

from qk_rope_analysis.constants import (
    DEFAULT_AIME2025_PROMPT_FIELD,
    DEFAULT_AIME2025_PROMPT_FILE,
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
)
from qk_rope_analysis.workflow import analyze_captured_tensors, export_analysis_artifacts, prepare_run_context


DEFAULT_LOCAL_MODEL = Path(__file__).resolve().parent / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture pre-RoPE Q/K and plot frequency-band clouds.")
    parser.add_argument("--model", default=str(DEFAULT_LOCAL_MODEL), help="Hugging Face model id or local model path.")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Inline calibration text. If unchanged and --prompt-file is omitted, AIME2025 is used.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=f"Text/JSON/JSONL calibration file. Defaults to {DEFAULT_AIME2025_PROMPT_FILE} and downloads if missing.",
    )
    parser.add_argument(
        "--prompt-field",
        default=None,
        help=f"JSON/JSONL text field. The default AIME2025 file uses {DEFAULT_AIME2025_PROMPT_FIELD!r}.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--device",
        default="auto",
        help='Device or comma-separated preference list: "auto", "cpu", "cuda", "cuda:1,auto", ...',
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--layers", default="0", help='Comma-separated layer indices, or "all".')
    parser.add_argument("--heads", default="0", help='Comma-separated query head indices, or "all".')
    parser.add_argument("--max-length", type=int, default=10000, help="Maximum tokenized calibration length.")
    parser.add_argument("--output-dir", default=None, help="Directory for metadata, CSV, and plots.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--plot-top-bands", type=int, default=2, help="Top-K dominant frequency bands.")
    parser.add_argument("--plot-max-points", type=int, default=2000, help="Max sampled points per Q or K cloud.")
    parser.add_argument(
        "--plot-radius-quantile",
        type=float,
        default=0.995,
        help="Axis limit coordinate quantile. Plots still expand to include the full coordinate range.",
    )
    parser.add_argument("--save-complex-tensors", action="store_true", help="Save complex_pairs.pt.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = prepare_run_context(args)
    print(f"Selected device: {context.metadata['device']}")
    artifacts = analyze_captured_tensors(context)
    export_analysis_artifacts(args, context, artifacts)
    print(f"Saved artifacts to: {context.output_dir}")


if __name__ == "__main__":
    main()

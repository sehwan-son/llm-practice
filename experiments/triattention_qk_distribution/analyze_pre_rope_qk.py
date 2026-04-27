import argparse
from pathlib import Path

from qk_rope_analysis.constants import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT
from qk_rope_analysis.reporting import print_analysis_report
from qk_rope_analysis.workflow import (
    analyze_captured_tensors,
    export_analysis_artifacts,
    prepare_run_context,
)


DEFAULT_LOCAL_MODEL = Path(__file__).resolve().parent / "models"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture pre-RoPE Q/K tensors and analyze RoPE pairs as complex-plane point clouds."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_LOCAL_MODEL),
        help="Hugging Face model id or local model path. Defaults to the local Qwen3-8B files under models/.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text used for the forward pass.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when the tokenizer has a chat template.",
    )
    parser.add_argument("--device", default="auto", help='Device to use: "auto", "cuda", "cuda:0", or "cpu".')
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--tensor",
        default="both",
        choices=["q", "k", "both"],
        help="Which pre-RoPE tensor to summarize/export. Figure2A plots always use both captured Q and K.",
    )
    parser.add_argument(
        "--layers",
        default="0",
        help='Comma-separated layer indices to capture, or "all".',
    )
    parser.add_argument(
        "--heads",
        default="0",
        help='Comma-separated head indices to analyze, or "all".',
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum prompt length after tokenization. Longer prompts are truncated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where summary artifacts will be written.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use the prompt as-is even if the tokenizer exposes a chat template.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create complex-plane plots for selected layers and heads.",
    )
    parser.add_argument(
        "--plot-mode",
        default="figure2a",
        choices=["figure2a", "pair-grid", "both"],
        help=(
            "Plot mode. figure2a overlays pre-RoPE Q/K at each head's dominant band; "
            "pair-grid keeps the older per-tensor all-pair grids."
        ),
    )
    parser.add_argument(
        "--plot-type",
        default="hist2d",
        choices=["hist2d", "scatter"],
        help="For plot-mode=pair-grid, plot each pair using a 2D histogram or sampled scatter plot.",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=80,
        help="Number of bins per axis when plot-type=hist2d.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=2000,
        help="Maximum number of points per cloud to draw in scatter plots.",
    )
    parser.add_argument(
        "--plot-top-bands",
        type=int,
        default=1,
        help="Number of dominant Q/K RoPE bands to show when plot-mode=figure2a.",
    )
    parser.add_argument(
        "--dominant-band-metric",
        default="center_product",
        choices=["center_product", "mean_abs_product"],
        help=(
            "How to select the dominant frequency band for Figure2A plots. "
            "center_product uses |mean(Q_band)| * |mean(K_band)|, the paper-style "
            "pre-RoPE center amplitude in the trigonometric series. "
            "mean_abs_product uses mean(|Q_band| * |K_band|)."
        ),
    )
    parser.add_argument(
        "--plot-radius-quantile",
        type=float,
        default=0.995,
        help="Axis limit is chosen from this quantile of absolute coordinates.",
    )
    parser.add_argument(
        "--save-complex-tensors",
        action="store_true",
        help="Save complex-pair tensors to complex_pairs.pt.",
    )
    parser.add_argument(
        "--plot-summary",
        action="store_true",
        help="Create layer/head summary plots from pre-RoPE band statistics.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export flat per-head and per-layer metrics as CSV files.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    context = prepare_run_context(args)
    artifacts = analyze_captured_tensors(args, context)
    export_analysis_artifacts(args, context, artifacts)
    print_analysis_report(output_dir=context.output_dir, tensor_names=context.tensor_names, summary=artifacts.summary)


if __name__ == "__main__":
    main()

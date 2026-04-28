import argparse
from pathlib import Path

from .constants import (
    DEFAULT_AIME2025_PROMPT_FIELD,
    DEFAULT_AIME2025_PROMPT_FILE,
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    EXPERIMENT_ROOT,
)


DEFAULT_LOCAL_MODEL = EXPERIMENT_ROOT / "models"


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
        help=f"Text/JSON/JSONL calibration file. Defaults to {Path(DEFAULT_AIME2025_PROMPT_FILE)} and downloads if missing.",
    )
    parser.add_argument(
        "--prompt-field",
        default=None,
        help=f"JSON/JSONL text field. The default AIME2025 file uses {DEFAULT_AIME2025_PROMPT_FIELD!r}.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--device",
        default="cuda:1,cuda:0",
        help='Device or comma-separated preference list: "cuda:1,cuda:0", "auto", "cpu", "cuda", ...',
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
    add_key_magnitude_plot_args(parser)
    parser.add_argument("--save-complex-tensors", action="store_true", help="Save complex_pairs.pt.")
    return parser


def add_key_magnitude_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key-magnitude-plot-kind",
        default="both",
        choices=["heatmap", "surface3d", "both"],
        help="Which pre-RoPE key magnitude plot to save.",
    )
    parser.add_argument(
        "--key-magnitude-max-tokens",
        type=int,
        default=0,
        help="Max token rows in pre-RoPE key magnitude heatmaps. Use 0 to plot every captured token.",
    )
    parser.add_argument(
        "--key-magnitude-max-channels",
        type=int,
        default=0,
        help="Max channel columns in pre-RoPE key magnitude heatmaps. Use 0 to plot every key channel.",
    )
    parser.add_argument(
        "--key-magnitude-color-quantile",
        type=float,
        default=0.999,
        help="Colorbar upper quantile for pre-RoPE key magnitude plots. Use 1.0 for the full max.",
    )
    parser.add_argument(
        "--key-magnitude-3d-max-tokens",
        type=int,
        default=420,
        help="Max token rows in 3D pre-RoPE key magnitude surfaces.",
    )
    parser.add_argument(
        "--key-magnitude-3d-max-channels",
        type=int,
        default=320,
        help="Max channel columns in 3D surfaces. Salient high-magnitude channels are preserved.",
    )
    parser.add_argument(
        "--key-magnitude-3d-elev",
        type=float,
        default=28.0,
        help="Elevation angle for 3D key magnitude surfaces.",
    )
    parser.add_argument(
        "--key-magnitude-3d-azim",
        type=float,
        default=-62.0,
        help="Azimuth angle for 3D key magnitude surfaces.",
    )

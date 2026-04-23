import argparse
import json
from pathlib import Path

from qk_rope_analysis.analysis import build_combined_metric_rows, resolve_tensor_names, write_csv
from qk_rope_analysis.plotting import maybe_plot_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create summary plots from a saved pre-RoPE Q/K summary.json.")
    parser.add_argument("--summary-json", required=True, help="Path to summary.json produced by analyze_pre_rope_qk.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where summary_plots and CSV exports will be written. Defaults to the summary.json parent.",
    )
    parser.add_argument(
        "--tensor",
        default="both",
        choices=["q", "k", "both"],
        help="Which tensor summaries to plot/export.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export head_metrics.csv and layer_metrics.csv from the loaded summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    tensor_names = resolve_tensor_names(args.tensor)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_plot_summary(summary=summary, tensor_names=tensor_names, output_dir=output_dir)

    if args.export_csv:
        head_rows, layer_rows = build_combined_metric_rows(summary, tensor_names)
        write_csv(output_dir / "head_metrics.csv", head_rows)
        write_csv(output_dir / "layer_metrics.csv", layer_rows)

    print(f"Saved summary plots to: {output_dir / 'summary_plots'}")
    if args.export_csv:
        print(f"Saved CSV exports to: {output_dir}")


if __name__ == "__main__":
    main()

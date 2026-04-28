from qk_rope_analysis.cli import build_parser
from qk_rope_analysis.workflow import analyze_captured_tensors, export_analysis_artifacts, prepare_run_context


def main() -> None:
    args = build_parser().parse_args()
    context = prepare_run_context(args)
    print(f"Selected device: {context.metadata['device']}")
    artifacts = analyze_captured_tensors(context)
    export_analysis_artifacts(args, context, artifacts)
    print(f"Saved artifacts to: {context.output_dir}")


if __name__ == "__main__":
    main()

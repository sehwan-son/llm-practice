from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .complex_pairs import to_rope_complex_pairs
from .config import (
    build_prompt_text,
    default_output_dir,
    parse_index_selection,
    resolve_device,
    resolve_dtype,
    resolve_tensor_names,
)
from .constants import (
    PAIRING_MODE,
    PAIRING_NOTE,
)
from .dominant_bands import build_qk_dominant_band_rows
from .metrics import build_combined_metric_rows
from .modeling import (
    capture_pre_rope_qk,
    get_decoder_layers,
    get_layer_rope_inv_freq,
    load_model,
    load_tokenizer,
)
from .plotting import maybe_plot_complex_pairs, maybe_plot_qk_dominant_bands, maybe_plot_summary
from .reporting import print_analysis_report
from .serialization import write_csv, write_json
from .summaries import summarize_complex_pairs

__all__ = [
    "AnalysisArtifacts",
    "PlotTask",
    "RunContext",
    "analyze_captured_tensors",
    "export_analysis_artifacts",
    "prepare_run_context",
    "print_analysis_report",
]


def should_plot_figure2a(args) -> bool:
    return bool(args.plot and args.plot_mode in {"figure2a", "both"})


def should_plot_pair_grid(args) -> bool:
    return bool(args.plot and args.plot_mode in {"pair-grid", "both"})


@dataclass
class RunContext:
    device: str
    dtype: torch.dtype
    layers: Any
    selected_layers: list[int]
    tensor_names: list[str]
    output_dir: Path
    metadata: dict[str, Any]
    captured: dict[str, dict[int, torch.Tensor]]


@dataclass
class PlotTask:
    complex_pairs: torch.Tensor
    tensor_name: str
    layer_idx: int
    selected_heads: list[int]


@dataclass
class AnalysisArtifacts:
    summary: dict[str, Any]
    complex_tensor_store: dict[str, dict[int, torch.Tensor]]
    plot_tasks: list[PlotTask]


def prepare_run_context(args) -> RunContext:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args.model, device=device, dtype=dtype, trust_remote_code=args.trust_remote_code)

    prompt_text = build_prompt_text(
        tokenizer,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        use_chat_template=not args.no_chat_template,
    )
    tokenized = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    model_inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}

    layers = get_decoder_layers(model)
    selected_layers = parse_index_selection(args.layers, len(layers), label="layer")
    captured = capture_pre_rope_qk(model, model_inputs=model_inputs, selected_layers=selected_layers)
    tensor_names = resolve_tensor_names(args.tensor)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "selected_layers": selected_layers,
        "tensor_selection": tensor_names,
        "token_count": int(model_inputs["input_ids"].shape[-1]),
        "prompt": args.prompt,
        "prompt_text_after_template": prompt_text,
        "token_ids": model_inputs["input_ids"][0].detach().cpu().tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0].detach().cpu().tolist()),
        "pairing_mode": PAIRING_MODE,
        "pairing_note": PAIRING_NOTE,
        "plot_mode": args.plot_mode if args.plot else None,
        "plot_top_bands": args.plot_top_bands if args.plot else None,
        "dominant_band_metric": args.dominant_band_metric if args.plot else None,
        "dominant_band_metric_note": (
            "center_product selects max |mean(Q_band)| * |mean(K_band)|, matching the "
            "pre-RoPE center amplitude used by the paper's trigonometric-series view; "
            "mean_abs_product selects max mean(|Q_band| * |K_band|)."
        ),
    }

    return RunContext(
        device=device,
        dtype=dtype,
        layers=layers,
        selected_layers=selected_layers,
        tensor_names=tensor_names,
        output_dir=output_dir,
        metadata=metadata,
        captured=captured,
    )


def analyze_captured_tensors(args, context: RunContext) -> AnalysisArtifacts:
    summary: dict[str, Any] = {}
    complex_tensor_store: dict[str, dict[int, torch.Tensor]] = {}
    plot_tasks: list[PlotTask] = []
    required_tensor_names = list(context.tensor_names)
    if should_plot_figure2a(args):
        required_tensor_names = sorted(set(required_tensor_names) | {"q", "k"})

    for tensor_name in context.tensor_names:
        summary[tensor_name] = {}

    for tensor_name in required_tensor_names:
        complex_tensor_store[tensor_name] = {}
        for layer_idx in context.selected_layers:
            raw_tensor = context.captured[tensor_name][layer_idx]
            complex_pairs = to_rope_complex_pairs(raw_tensor)
            layer_inv_freq = get_layer_rope_inv_freq(context.layers[layer_idx], num_pairs=complex_pairs.shape[-1])
            selected_heads = None
            needs_tensor_head_selection = tensor_name in context.tensor_names
            if needs_tensor_head_selection:
                selected_heads = parse_index_selection(args.heads, complex_pairs.shape[2], label=f"{tensor_name} head")

            if tensor_name in context.tensor_names:
                summary[tensor_name][str(layer_idx)] = summarize_complex_pairs(
                    raw_tensor=raw_tensor,
                    complex_pairs=complex_pairs,
                    selected_heads=selected_heads or [],
                    inv_freq=layer_inv_freq,
                )
            complex_tensor_store[tensor_name][layer_idx] = complex_pairs

            if should_plot_pair_grid(args) and tensor_name in context.tensor_names:
                plot_tasks.append(
                    PlotTask(
                        complex_pairs=complex_pairs,
                        tensor_name=tensor_name,
                        layer_idx=layer_idx,
                        selected_heads=selected_heads or [],
                    )
                )

    return AnalysisArtifacts(
        summary=summary,
        complex_tensor_store=complex_tensor_store,
        plot_tasks=plot_tasks,
    )


def export_analysis_artifacts(args, context: RunContext, artifacts: AnalysisArtifacts) -> None:
    write_json(context.output_dir / "metadata.json", context.metadata)
    write_json(context.output_dir / "summary.json", artifacts.summary)

    if args.save_complex_tensors:
        torch.save(artifacts.complex_tensor_store, context.output_dir / "complex_pairs.pt")

    if args.export_csv:
        head_rows, layer_rows = build_combined_metric_rows(artifacts.summary, context.tensor_names)
        write_csv(context.output_dir / "head_metrics.csv", head_rows)
        write_csv(context.output_dir / "layer_metrics.csv", layer_rows)

    if args.plot_summary:
        maybe_plot_summary(summary=artifacts.summary, tensor_names=context.tensor_names, output_dir=context.output_dir)

    if should_plot_figure2a(args):
        dominant_band_rows = []
        for layer_idx in context.selected_layers:
            q_complex_pairs = artifacts.complex_tensor_store["q"][layer_idx]
            k_complex_pairs = artifacts.complex_tensor_store["k"][layer_idx]
            selected_query_heads = parse_index_selection(
                args.heads,
                q_complex_pairs.shape[2],
                label="query head",
            )
            dominant_band_rows.extend(
                build_qk_dominant_band_rows(
                    q_complex_pairs=q_complex_pairs,
                    k_complex_pairs=k_complex_pairs,
                    layer_idx=layer_idx,
                    selected_query_heads=selected_query_heads,
                    top_bands=args.plot_top_bands,
                    metric=args.dominant_band_metric,
                )
            )
            maybe_plot_qk_dominant_bands(
                q_complex_pairs=q_complex_pairs,
                k_complex_pairs=k_complex_pairs,
                layer_idx=layer_idx,
                selected_query_heads=selected_query_heads,
                output_dir=context.output_dir,
                plot_max_points=args.plot_max_points,
                plot_radius_quantile=args.plot_radius_quantile,
                top_bands=args.plot_top_bands,
                dominant_band_metric=args.dominant_band_metric,
            )
        write_csv(
            context.output_dir / f"figure2a_dominant_bands_{args.dominant_band_metric}.csv",
            dominant_band_rows,
        )

    if should_plot_pair_grid(args):
        for task in artifacts.plot_tasks:
            maybe_plot_complex_pairs(
                complex_pairs=task.complex_pairs,
                tensor_name=task.tensor_name,
                layer_idx=task.layer_idx,
                selected_heads=task.selected_heads,
                output_dir=context.output_dir,
                plot_type=args.plot_type,
                plot_bins=args.plot_bins,
                plot_max_points=args.plot_max_points,
                plot_radius_quantile=args.plot_radius_quantile,
            )

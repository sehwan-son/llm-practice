from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .analysis import (
    PAIRING_MODE,
    PAIRING_NOTE,
    build_combined_metric_rows,
    build_prompt_text,
    capture_pre_rope_qk,
    default_output_dir,
    get_decoder_layers,
    get_layer_rope_inv_freq,
    load_model,
    load_tokenizer,
    parse_index_selection,
    resolve_device,
    resolve_dtype,
    resolve_tensor_names,
    summarize_complex_pairs,
    to_rope_complex_pairs,
    write_csv,
    write_json,
)
from .plotting import maybe_plot_complex_pairs, maybe_plot_summary


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

    for tensor_name in context.tensor_names:
        summary[tensor_name] = {}
        complex_tensor_store[tensor_name] = {}
        for layer_idx in context.selected_layers:
            raw_tensor = context.captured[tensor_name][layer_idx]
            complex_pairs = to_rope_complex_pairs(raw_tensor)
            selected_heads = parse_index_selection(args.heads, complex_pairs.shape[2], label=f"{tensor_name} head")
            layer_inv_freq = get_layer_rope_inv_freq(context.layers[layer_idx], num_pairs=complex_pairs.shape[-1])

            summary[tensor_name][str(layer_idx)] = summarize_complex_pairs(
                raw_tensor=raw_tensor,
                complex_pairs=complex_pairs,
                selected_heads=selected_heads,
                inv_freq=layer_inv_freq,
            )
            complex_tensor_store[tensor_name][layer_idx] = complex_pairs

            if args.plot:
                plot_tasks.append(
                    PlotTask(
                        complex_pairs=complex_pairs,
                        tensor_name=tensor_name,
                        layer_idx=layer_idx,
                        selected_heads=selected_heads,
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

    if args.plot:
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


def format_dominant_bands(head_summary: dict[str, Any]) -> str:
    band_dist = head_summary["frequency_band_distribution"]
    dominant_bands = []
    for pair_summary in band_dist["dominant_pairs"][:3]:
        band_label = f"pair {pair_summary['pair_idx']} ({pair_summary['energy_share'] * 100:.1f}%"
        if "wavelength_tokens" in pair_summary:
            band_label += f", lambda={pair_summary['wavelength_tokens']:.1f}"
        dominant_bands.append(f"{band_label})")
    return ", ".join(dominant_bands)


def print_analysis_report(output_dir: Path, tensor_names: list[str], summary: dict[str, Any]) -> None:
    print(f"Saved artifacts to: {output_dir}")
    print(f"RoPE pairing note: {PAIRING_NOTE}")

    for tensor_name in tensor_names:
        print(f"\n[{tensor_name.upper()}] pre-RoPE head distribution")
        for layer_key, layer_summary in summary[tensor_name].items():
            for head_key, head_summary in layer_summary["per_head"].items():
                aggregate = head_summary["aggregate"]
                raw_dist = head_summary["raw_head_distribution"]
                band_dist = head_summary["frequency_band_distribution"]
                print(
                    f"  layer {layer_key} head {head_key}: "
                    f"value_std={raw_dist['value']['std']:.4f}, "
                    f"vector_l2_mean={raw_dist['vector_l2_norm']['mean']:.4f}, "
                    f"top1_band_share={band_dist['top1_energy_share']:.4f}, "
                    f"top4_band_share={band_dist['top4_energy_share']:.4f}, "
                    f"band_entropy={band_dist['normalized_entropy']:.4f}, "
                    f"center_radius={aggregate['center_radius']:.4f}, "
                    f"mean_radius={aggregate['mean_radius']:.4f}, "
                    f"major_axis_std={aggregate['major_axis_std']:.4f}, "
                    f"minor_axis_std={aggregate['minor_axis_std']:.4f}, "
                    f"axis_ratio={aggregate['axis_ratio']:.4f}, "
                    f"dominant_bands=[{format_dominant_bands(head_summary)}]"
                )

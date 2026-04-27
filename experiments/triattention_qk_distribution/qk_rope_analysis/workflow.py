from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .complex_pairs import to_rope_complex_pairs
from .config import (
    build_prompt_text,
    default_output_dir,
    load_calibration_prompt,
    parse_index_selection,
    resolve_device,
    resolve_dtype,
)
from .constants import (
    DEFAULT_AIME2025_DOWNLOAD_URL,
    DEFAULT_AIME2025_PROMPT_FIELD,
    DEFAULT_AIME2025_PROMPT_FILE,
    DEFAULT_PROMPT,
    PAIRING_MODE,
    PAIRING_NOTE,
)
from .dominant_bands import build_qk_dominant_band_rows
from .modeling import capture_pre_rope_qk, get_decoder_layers, get_layer_rope_inv_freq, load_model, load_tokenizer
from .plotting import plot_qk_frequency_grids, plot_qk_top_frequency_bands
from .serialization import write_csv, write_json


@dataclass
class RunContext:
    layers: Any
    selected_layers: list[int]
    output_dir: Path
    metadata: dict[str, Any]
    captured: dict[str, dict[int, torch.Tensor]]


@dataclass
class AnalysisArtifacts:
    complex_pairs: dict[str, dict[int, torch.Tensor]]


def resolve_calibration_source(args) -> tuple[str | None, str | None, str | None]:
    if args.prompt_file:
        return args.prompt_file, args.prompt_field, None
    if args.prompt != DEFAULT_PROMPT:
        return None, args.prompt_field, None
    return (
        str(DEFAULT_AIME2025_PROMPT_FILE),
        args.prompt_field or DEFAULT_AIME2025_PROMPT_FIELD,
        DEFAULT_AIME2025_DOWNLOAD_URL,
    )


def prepare_run_context(args) -> RunContext:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args.model, device=device, dtype=dtype, trust_remote_code=args.trust_remote_code)

    prompt_file, prompt_field, download_url = resolve_calibration_source(args)
    calibration_prompt = load_calibration_prompt(args.prompt, prompt_file, prompt_field, download_url)
    prompt_text = build_prompt_text(
        tokenizer,
        prompt=calibration_prompt,
        system_prompt=args.system_prompt,
        use_chat_template=not args.no_chat_template,
    )
    tokenized = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length)
    model_inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}

    layers = get_decoder_layers(model)
    selected_layers = parse_index_selection(args.layers, len(layers), label="layer")
    captured = capture_pre_rope_qk(model, model_inputs=model_inputs, selected_layers=selected_layers)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "selected_layers": selected_layers,
        "selected_heads": args.heads,
        "token_count": int(model_inputs["input_ids"].shape[-1]),
        "prompt_source": str(Path(prompt_file).expanduser().resolve()) if prompt_file else "inline",
        "prompt_field": prompt_field,
        "prompt_download_url": download_url,
        "prompt_text_after_template": prompt_text,
        "pairing_mode": PAIRING_MODE,
        "pairing_note": PAIRING_NOTE,
        "dominant_band_metric": "expected_norm_product",
        "dominant_band_metric_note": "C_f = E[|q_f|] * E[|k_f|], TriAttention Appendix B.7 Eq. 26",
        "plot_top_bands": args.plot_top_bands,
    }
    return RunContext(layers, selected_layers, output_dir, metadata, captured)


def analyze_captured_tensors(context: RunContext) -> AnalysisArtifacts:
    complex_pairs = {"q": {}, "k": {}}
    for tensor_name in ("q", "k"):
        for layer_idx in context.selected_layers:
            complex_pairs[tensor_name][layer_idx] = to_rope_complex_pairs(context.captured[tensor_name][layer_idx])
    return AnalysisArtifacts(complex_pairs)


def export_analysis_artifacts(args, context: RunContext, artifacts: AnalysisArtifacts) -> None:
    dominant_band_rows = []
    for layer_idx in context.selected_layers:
        q_complex_pairs = artifacts.complex_pairs["q"][layer_idx]
        k_complex_pairs = artifacts.complex_pairs["k"][layer_idx]
        inv_freq = get_layer_rope_inv_freq(context.layers[layer_idx], num_pairs=q_complex_pairs.shape[-1])
        selected_heads = parse_index_selection(args.heads, q_complex_pairs.shape[2], label="query head")

        dominant_band_rows.extend(
            build_qk_dominant_band_rows(
                q_complex_pairs=q_complex_pairs,
                k_complex_pairs=k_complex_pairs,
                layer_idx=layer_idx,
                selected_query_heads=selected_heads,
                top_bands=args.plot_top_bands,
                inv_freq=inv_freq,
            )
        )
        plot_qk_frequency_grids(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            layer_idx=layer_idx,
            selected_query_heads=selected_heads,
            output_dir=context.output_dir,
            plot_max_points=args.plot_max_points,
            plot_radius_quantile=args.plot_radius_quantile,
            top_bands=args.plot_top_bands,
            inv_freq=inv_freq,
        )
        plot_qk_top_frequency_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            layer_idx=layer_idx,
            selected_query_heads=selected_heads,
            output_dir=context.output_dir,
            plot_max_points=args.plot_max_points,
            plot_radius_quantile=args.plot_radius_quantile,
            top_bands=args.plot_top_bands,
            inv_freq=inv_freq,
        )

    write_json(context.output_dir / "metadata.json", context.metadata)
    write_csv(context.output_dir / "dominant_frequency_bands.csv", dominant_band_rows)
    if args.save_complex_tensors:
        torch.save(artifacts.complex_pairs, context.output_dir / "complex_pairs.pt")

import csv
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = "한국어로 자기소개를 두 문장으로 해줘."
DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful assistant."
PAIRING_MODE = "split_half"
PAIR_DEFINITION = "complex_pair[p] = x[p] + i * x[p + head_dim/2]"
PAIRING_NOTE = "Qwen3 rotate_half pairs dim i with dim i + head_dim/2, not adjacent dims."
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_tensor_names(selection: str) -> list[str]:
    return ["q", "k"] if selection == "both" else [selection]


def sanitize_model_name(model_name: str) -> str:
    return model_name.strip("/").replace("/", "__")


def default_output_dir(model_name: str) -> Path:
    base_dir = EXPERIMENT_ROOT / "outputs" / "runs"
    return base_dir / sanitize_model_name(model_name)


def build_prompt_text(tokenizer, prompt: str, system_prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_tokenizer(model_name: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=trust_remote_code)


def load_model(model_name: str, device: str, dtype: torch.dtype, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def get_decoder_layers(model) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Could not find decoder layers at model.model.layers.")
    return layers


def parse_index_selection(selection_arg: str, limit: int, label: str) -> list[int]:
    if selection_arg == "all":
        return list(range(limit))

    selected = []
    for raw_item in selection_arg.split(","):
        item = raw_item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= limit:
            raise ValueError(f"{label} index {index} is out of range for size {limit}.")
        selected.append(index)

    if not selected:
        raise ValueError(f"No valid {label} indices were selected.")
    return sorted(set(selected))


def make_capture_hook(store: dict[str, dict[int, torch.Tensor]], tensor_name: str, layer_idx: int):
    def hook(_module, _inputs, output):
        store[tensor_name][layer_idx] = output.detach().to(device="cpu", dtype=torch.float32).contiguous()

    return hook


def capture_pre_rope_qk(
    model,
    model_inputs: dict[str, torch.Tensor],
    selected_layers: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    layers = get_decoder_layers(model)
    captured = {"q": {}, "k": {}}

    with ExitStack() as stack:
        for layer_idx in selected_layers:
            self_attn = layers[layer_idx].self_attn
            if not hasattr(self_attn, "q_norm") or not hasattr(self_attn, "k_norm"):
                raise ValueError(
                    f"Layer {layer_idx} does not expose q_norm/k_norm. "
                    "This script expects a Qwen3-style attention module."
                )

            q_handle = self_attn.q_norm.register_forward_hook(make_capture_hook(captured, "q", layer_idx))
            k_handle = self_attn.k_norm.register_forward_hook(make_capture_hook(captured, "k", layer_idx))
            stack.callback(q_handle.remove)
            stack.callback(k_handle.remove)

        with torch.inference_mode():
            model(**model_inputs, use_cache=False)

    return captured


def to_rope_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [batch, seq, heads, head_dim], got {tuple(tensor.shape)}.")
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    half = head_dim // 2
    real = tensor[..., :half]
    imag = tensor[..., half:]
    return torch.complex(real, imag)


def map_query_head_to_key_head(query_head_idx: int, num_query_heads: int, num_key_heads: int) -> int:
    if num_query_heads == num_key_heads:
        return query_head_idx
    if num_query_heads % num_key_heads != 0:
        raise ValueError(
            f"Cannot map {num_query_heads} query heads to {num_key_heads} key heads with an integer GQA ratio."
        )
    return query_head_idx // (num_query_heads // num_key_heads)


def mean_resultant_length(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("Cannot compute mean resultant length for an empty complex cloud.")

    values = values.to(torch.complex64).reshape(-1)
    mean_radius = torch.abs(values).mean()
    if mean_radius.item() <= 0:
        return 0.0
    return float((torch.abs(values.mean()) / mean_radius).item())


def qk_band_contribution_scores(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    query_head_idx: int,
) -> tuple[torch.Tensor, int]:
    if q_complex_pairs.shape[:2] != k_complex_pairs.shape[:2]:
        raise ValueError(
            "Q and K complex tensors must share batch/sequence shape, "
            f"got {tuple(q_complex_pairs.shape[:2])} and {tuple(k_complex_pairs.shape[:2])}."
        )
    if q_complex_pairs.shape[-1] != k_complex_pairs.shape[-1]:
        raise ValueError(
            f"Q and K must have the same number of RoPE pairs, got {q_complex_pairs.shape[-1]} "
            f"and {k_complex_pairs.shape[-1]}."
        )

    num_query_heads = q_complex_pairs.shape[2]
    num_key_heads = k_complex_pairs.shape[2]
    key_head_idx = map_query_head_to_key_head(query_head_idx, num_query_heads, num_key_heads)

    q_values = q_complex_pairs[:, :, query_head_idx, :].to(torch.complex64)
    k_values = k_complex_pairs[:, :, key_head_idx, :].to(torch.complex64)
    scores = (torch.abs(q_values) * torch.abs(k_values)).mean(dim=(0, 1)).to(torch.float32)
    return scores, key_head_idx


def select_dominant_qk_bands(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    query_head_idx: int,
    top_k: int,
) -> tuple[list[int], torch.Tensor, int]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    scores, key_head_idx = qk_band_contribution_scores(
        q_complex_pairs=q_complex_pairs,
        k_complex_pairs=k_complex_pairs,
        query_head_idx=query_head_idx,
    )
    top_count = min(top_k, scores.numel())
    _, indices = torch.topk(scores, k=top_count, largest=True, sorted=True)
    return indices.tolist(), scores, key_head_idx


def get_layer_rope_inv_freq(layer, num_pairs: int) -> torch.Tensor | None:
    rotary_emb = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None:
        return None

    inv_freq = inv_freq.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if inv_freq.numel() != num_pairs:
        return None
    return inv_freq


def flatten_complex_cloud(complex_pairs: torch.Tensor, head_idx: int, pair_idx: int | None = None) -> torch.Tensor:
    head_values = complex_pairs[:, :, head_idx, :]
    if pair_idx is None:
        return head_values.reshape(-1)
    return head_values[:, :, pair_idx].reshape(-1)


def summarize_real_values(values: torch.Tensor) -> dict[str, float]:
    quantile_points = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=values.dtype)
    quantiles = torch.quantile(values, quantile_points)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
    }


def summarize_complex_cloud(values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        raise ValueError("Cannot summarize an empty complex cloud.")

    real = values.real.to(torch.float32)
    imag = values.imag.to(torch.float32)
    radius = torch.abs(values).to(torch.float32)

    center_real = real.mean()
    center_imag = imag.mean()
    centered = torch.stack([real - center_real, imag - center_imag], dim=1)
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    major = float(torch.sqrt(eigvals[-1]).item())
    minor = float(torch.sqrt(eigvals[0]).item())
    axis_ratio = float(major / max(minor, 1e-12))

    return {
        "num_points": int(values.numel()),
        "center_real": float(center_real.item()),
        "center_imag": float(center_imag.item()),
        "center_radius": float(torch.sqrt(center_real.square() + center_imag.square()).item()),
        "mean_resultant_length": mean_resultant_length(values),
        "mean_radius": float(radius.mean().item()),
        "std_radius": float(radius.std(unbiased=False).item()),
        "rms_radius": float(torch.sqrt((radius.square()).mean()).item()),
        "major_axis_std": major,
        "minor_axis_std": minor,
        "axis_ratio": axis_ratio,
        "real": summarize_real_values(real),
        "imag": summarize_real_values(imag),
        "radius": summarize_real_values(radius),
    }


def summarize_head_value_distribution(head_tensor: torch.Tensor) -> dict[str, Any]:
    if head_tensor.ndim != 3:
        raise ValueError(f"Expected [batch, seq, head_dim], got {tuple(head_tensor.shape)}.")

    head_tensor = head_tensor.to(torch.float32)
    flat_values = head_tensor.reshape(-1)
    abs_values = flat_values.abs()
    vector_norms = torch.linalg.vector_norm(head_tensor, dim=-1).reshape(-1)

    return {
        "shape": list(head_tensor.shape),
        "num_tokens": int(head_tensor.shape[0] * head_tensor.shape[1]),
        "head_dim": int(head_tensor.shape[-1]),
        "value": summarize_real_values(flat_values),
        "abs_value": summarize_real_values(abs_values),
        "value_rms": float(torch.sqrt(flat_values.square().mean()).item()),
        "mean_abs_value": float(abs_values.mean().item()),
        "vector_l2_norm": summarize_real_values(vector_norms),
    }


def summarize_frequency_band_distribution(
    complex_pairs: torch.Tensor,
    head_idx: int,
    inv_freq: torch.Tensor | None = None,
) -> dict[str, Any]:
    head_values = complex_pairs[:, :, head_idx, :].to(torch.complex64)
    num_pairs = head_values.shape[-1]
    radius = torch.abs(head_values).to(torch.float32)
    pair_energy = radius.square().mean(dim=(0, 1))

    total_energy = pair_energy.sum()
    if total_energy.item() <= 0:
        energy_share = torch.zeros_like(pair_energy)
    else:
        energy_share = pair_energy / total_energy

    sorted_share, sorted_idx = torch.sort(energy_share, descending=True)
    entropy = -(energy_share * torch.log(energy_share.clamp_min(1e-12))).sum()
    normalized_entropy = 0.0
    if num_pairs > 1:
        normalized_entropy = float((entropy / math.log(num_pairs)).item())

    per_pair = []
    rank_by_pair = torch.empty(num_pairs, dtype=torch.int64)
    rank_by_pair[sorted_idx] = torch.arange(num_pairs, dtype=torch.int64)
    for pair_idx in range(num_pairs):
        pair_summary = {
            "pair_idx": pair_idx,
            "dims": [pair_idx, pair_idx + num_pairs],
            "mean_energy": float(pair_energy[pair_idx].item()),
            "energy_share": float(energy_share[pair_idx].item()),
            "mean_radius": float(radius[:, :, pair_idx].mean().item()),
            "rms_radius": float(torch.sqrt(pair_energy[pair_idx]).item()),
            "rank_by_energy": int(rank_by_pair[pair_idx].item()) + 1,
        }
        if inv_freq is not None:
            freq = float(inv_freq[pair_idx].item())
            pair_summary["inv_freq"] = freq
            if freq > 0:
                pair_summary["wavelength_tokens"] = float((2 * math.pi) / freq)
        per_pair.append(pair_summary)

    def topk_share(k: int) -> float:
        return float(sorted_share[: min(k, num_pairs)].sum().item())

    dominant_pairs = [per_pair[pair_idx] for pair_idx in sorted_idx[: min(4, num_pairs)].tolist()]
    return {
        "num_pairs": int(num_pairs),
        "energy_metric": "mean(|complex_pair|^2) across batch and sequence positions",
        "top1_energy_share": topk_share(1),
        "top2_energy_share": topk_share(2),
        "top4_energy_share": topk_share(4),
        "top8_energy_share": topk_share(8),
        "normalized_entropy": normalized_entropy,
        "effective_num_bands": float(math.exp(entropy.item())),
        "dominant_pairs": dominant_pairs,
        "per_pair": per_pair,
    }


def summarize_complex_pairs(
    raw_tensor: torch.Tensor,
    complex_pairs: torch.Tensor,
    selected_heads: list[int],
    inv_freq: torch.Tensor | None = None,
) -> dict[str, Any]:
    num_pairs = complex_pairs.shape[-1]
    per_head = {}
    for head_idx in selected_heads:
        raw_head_tensor = raw_tensor[:, :, head_idx, :]
        aggregate_values = flatten_complex_cloud(complex_pairs, head_idx=head_idx, pair_idx=None)
        pair_summaries = []
        for pair_idx in range(num_pairs):
            pair_values = flatten_complex_cloud(complex_pairs, head_idx=head_idx, pair_idx=pair_idx)
            pair_summaries.append(
                {
                    "pair_idx": pair_idx,
                    "dims": [pair_idx, pair_idx + num_pairs],
                    **summarize_complex_cloud(pair_values),
                }
            )

        per_head[str(head_idx)] = {
            "raw_head_distribution": summarize_head_value_distribution(raw_head_tensor),
            "frequency_band_distribution": summarize_frequency_band_distribution(
                complex_pairs,
                head_idx=head_idx,
                inv_freq=inv_freq,
            ),
            "aggregate": summarize_complex_cloud(aggregate_values),
            "per_pair": pair_summaries,
        }

    return {
        "raw_shape": list(raw_tensor.shape),
        "shape": list(complex_pairs.shape),
        "num_heads": int(complex_pairs.shape[2]),
        "num_pairs": int(num_pairs),
        "pairing_mode": PAIRING_MODE,
        "pair_definition": PAIR_DEFINITION,
        "selected_heads": selected_heads,
        "per_head": per_head,
    }


def flatten_head_metric_rows(summary: dict[str, Any], tensor_name: str) -> list[dict[str, Any]]:
    rows = []
    for layer_key in sorted(summary.keys(), key=int):
        layer_summary = summary[layer_key]
        for head_key in sorted(layer_summary["per_head"].keys(), key=int):
            head_summary = layer_summary["per_head"][head_key]
            raw_dist = head_summary["raw_head_distribution"]
            band_dist = head_summary["frequency_band_distribution"]
            aggregate = head_summary["aggregate"]
            dominant_pair = band_dist["dominant_pairs"][0] if band_dist["dominant_pairs"] else None

            rows.append(
                {
                    "tensor": tensor_name,
                    "layer": int(layer_key),
                    "head": int(head_key),
                    "head_dim": int(raw_dist["head_dim"]),
                    "num_pairs": int(layer_summary["num_pairs"]),
                    "value_std": float(raw_dist["value"]["std"]),
                    "value_rms": float(raw_dist["value_rms"]),
                    "mean_abs_value": float(raw_dist["mean_abs_value"]),
                    "vector_l2_mean": float(raw_dist["vector_l2_norm"]["mean"]),
                    "vector_l2_std": float(raw_dist["vector_l2_norm"]["std"]),
                    "top1_band_share": float(band_dist["top1_energy_share"]),
                    "top2_band_share": float(band_dist["top2_energy_share"]),
                    "top4_band_share": float(band_dist["top4_energy_share"]),
                    "top8_band_share": float(band_dist["top8_energy_share"]),
                    "band_entropy": float(band_dist["normalized_entropy"]),
                    "effective_num_bands": float(band_dist["effective_num_bands"]),
                    "center_radius": float(aggregate["center_radius"]),
                    "mean_radius": float(aggregate["mean_radius"]),
                    "major_axis_std": float(aggregate["major_axis_std"]),
                    "minor_axis_std": float(aggregate["minor_axis_std"]),
                    "axis_ratio": float(aggregate["axis_ratio"]),
                    "dominant_pair_idx": int(dominant_pair["pair_idx"]) if dominant_pair else -1,
                    "dominant_pair_share": float(dominant_pair["energy_share"]) if dominant_pair else 0.0,
                    "dominant_pair_wavelength_tokens": (
                        float(dominant_pair["wavelength_tokens"])
                        if dominant_pair and "wavelength_tokens" in dominant_pair
                        else float("nan")
                    ),
                }
            )
    return rows


def summarize_layer_metric_rows(head_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in head_rows:
        grouped.setdefault((row["tensor"], row["layer"]), []).append(row)

    layer_rows = []
    for (tensor_name, layer_idx), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        dominant_pair_counts: dict[int, int] = {}
        for row in rows:
            dominant_pair_counts[row["dominant_pair_idx"]] = dominant_pair_counts.get(row["dominant_pair_idx"], 0) + 1

        dominant_pair_mode = max(dominant_pair_counts.items(), key=lambda item: item[1])[0]
        dominant_pair_mode_fraction = dominant_pair_counts[dominant_pair_mode] / max(len(rows), 1)
        layer_rows.append(
            {
                "tensor": tensor_name,
                "layer": layer_idx,
                "num_heads": len(rows),
                "mean_top1_band_share": sum(row["top1_band_share"] for row in rows) / len(rows),
                "max_top1_band_share": max(row["top1_band_share"] for row in rows),
                "mean_top4_band_share": sum(row["top4_band_share"] for row in rows) / len(rows),
                "mean_band_entropy": sum(row["band_entropy"] for row in rows) / len(rows),
                "mean_effective_num_bands": sum(row["effective_num_bands"] for row in rows) / len(rows),
                "mean_vector_l2": sum(row["vector_l2_mean"] for row in rows) / len(rows),
                "mean_axis_ratio": sum(row["axis_ratio"] for row in rows) / len(rows),
                "dominant_pair_mode": dominant_pair_mode,
                "dominant_pair_mode_fraction": dominant_pair_mode_fraction,
            }
        )
    return layer_rows


def build_combined_metric_rows(
    summary: dict[str, Any],
    tensor_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    head_rows = []
    for tensor_name in tensor_names:
        head_rows.extend(flatten_head_metric_rows(summary[tensor_name], tensor_name=tensor_name))
    return head_rows, summarize_layer_metric_rows(head_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

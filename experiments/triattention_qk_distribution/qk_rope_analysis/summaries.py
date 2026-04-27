import math
from typing import Any

import torch

from .complex_pairs import flatten_complex_cloud, mean_resultant_length
from .constants import PAIR_DEFINITION, PAIRING_MODE


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

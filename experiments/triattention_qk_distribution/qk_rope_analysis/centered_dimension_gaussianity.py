import math
from pathlib import Path
from typing import Any

import torch

from .normality import format_pvalue, plot_hist_with_gaussian, plot_normal_qq, summarize_univariate_normality
from .plotting_common import load_matplotlib_pyplot


def mean_center_pre_rope_vectors(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [batch, seq, heads, head_dim], got {tuple(tensor.shape)}.")

    vectors = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, tensor.shape[-1])
    finite_mask = torch.isfinite(vectors)
    finite_counts = finite_mask.sum(dim=0)
    finite_values = torch.where(finite_mask, vectors, torch.zeros_like(vectors))
    mean_vector = finite_values.sum(dim=0) / finite_counts.clamp_min(1)
    mean_vector = torch.where(finite_counts > 0, mean_vector, torch.full_like(mean_vector, float("nan")))
    return vectors - mean_vector[None, :], mean_vector, finite_counts


def build_centered_dimension_gaussianity_rows(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    layer_idx: int,
) -> list[dict[str, Any]]:
    rows = []
    for tensor_name, tensor in (("q", q_tensor), ("k", k_tensor)):
        batch_size, token_count, head_count, head_dim = tensor.shape
        centered_vectors, mean_vector, finite_counts = mean_center_pre_rope_vectors(tensor)
        vector_count = int(centered_vectors.shape[0])

        for dim_idx in range(head_dim):
            stats = summarize_univariate_normality(centered_vectors[:, dim_idx])
            rows.append(
                {
                    "layer": layer_idx,
                    "tensor": tensor_name,
                    "dim": dim_idx,
                    "batch_size": int(batch_size),
                    "token_count": int(token_count),
                    "head_count": int(head_count),
                    "vector_count": vector_count,
                    "finite_count_for_mean": int(finite_counts[dim_idx].item()),
                    "subtracted_mean": float(mean_vector[dim_idx].item()),
                    "centered_n": int(stats["n"]),
                    "centered_mean": float(stats["mean"]),
                    "centered_std": float(stats["std"]),
                    "centered_skewness": float(stats["skewness"]),
                    "centered_excess_kurtosis": float(stats["excess_kurtosis"]),
                    "centered_jarque_bera": float(stats["jarque_bera"]),
                    "centered_jarque_bera_pvalue": float(stats["jarque_bera_pvalue"]),
                }
            )
    return rows


def _rank_dimensions_by_non_gaussianity(centered_vectors: torch.Tensor, top_dims: int) -> list[int]:
    if top_dims <= 0:
        return []

    scored_dims = []
    for dim_idx in range(centered_vectors.shape[-1]):
        stats = summarize_univariate_normality(centered_vectors[:, dim_idx])
        pvalue = float(stats["jarque_bera_pvalue"])
        score = pvalue if math.isfinite(pvalue) else float("inf")
        scored_dims.append((score, dim_idx))
    scored_dims.sort()
    return [dim_idx for _, dim_idx in scored_dims[: min(top_dims, len(scored_dims))]]


def plot_centered_dimension_gaussianity_diagnostics(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    layer_idx: int,
    output_dir: Path,
    plot_top_dims: int,
    max_points: int,
    hist_bins: int,
) -> None:
    if plot_top_dims <= 0:
        return

    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "centered_dimension_gaussianity"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for tensor_name, tensor, color in (("q", q_tensor, "tab:blue"), ("k", k_tensor, "tab:orange")):
        centered_vectors, mean_vector, _finite_counts = mean_center_pre_rope_vectors(tensor)
        dim_indices = _rank_dimensions_by_non_gaussianity(centered_vectors, plot_top_dims)
        if not dim_indices:
            continue

        fig, axes = plt.subplots(
            nrows=len(dim_indices),
            ncols=2,
            figsize=(11.5, max(2.8 * len(dim_indices), 3.2)),
            squeeze=False,
        )
        for row_idx, dim_idx in enumerate(dim_indices):
            values = centered_vectors[:, dim_idx]
            stats = summarize_univariate_normality(values)
            title = (
                f"{tensor_name.upper()} dim {dim_idx} centered histogram "
                f"(subtracted mean={float(mean_vector[dim_idx].item()):.3g})"
            )
            plot_hist_with_gaussian(
                axes[row_idx][0],
                values,
                title,
                color,
                max_points=max_points,
                bins=hist_bins,
            )
            plot_normal_qq(
                axes[row_idx][1],
                values,
                f"{tensor_name.upper()} dim {dim_idx} centered QQ",
                color,
                max_points=max_points,
            )
            axes[row_idx][1].text(
                0.98,
                0.04,
                f"skew={float(stats['skewness']):.2f}\n"
                f"kurt={float(stats['excess_kurtosis']):.2f}\n"
                f"JB p={format_pvalue(float(stats['jarque_bera_pvalue']))}",
                transform=axes[row_idx][1].transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )

        fig.suptitle(
            f"Mean-centered pre-RoPE {tensor_name.upper()} dimension Gaussianity, layer {layer_idx}\n"
            "Vectors are flattened over batch, token, and head before subtracting the per-dimension mean."
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(plot_dir / f"{tensor_name}_layer{layer_idx}_centered_dimension_gaussianity.png", dpi=180)
        plt.close(fig)

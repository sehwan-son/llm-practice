import math
from pathlib import Path
from typing import Any

import torch

from .dominant_bands import qk_band_contribution_scores, select_dominant_qk_bands
from .normality import plot_hist_with_gaussian, plot_normal_qq, summarize_complex_gaussianity
from .plotting_common import load_matplotlib_pyplot


def _selected_gaussianity_bands(num_pairs: int, ranked_bands: list[int], band_scope: str) -> list[int]:
    if band_scope == "all":
        return list(range(num_pairs))
    if band_scope == "top":
        return ranked_bands
    if band_scope == "none":
        return []
    raise ValueError(f"Unsupported gaussianity band scope: {band_scope!r}.")


def build_qk_gaussianity_rows(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    band_scope: str,
    top_bands: int,
    inv_freq: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows = []
    num_pairs = q_complex_pairs.shape[-1]
    top_bands = max(top_bands, 1)

    for query_head_idx in selected_query_heads:
        band_scores, key_head_idx = qk_band_contribution_scores(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
        )
        total_score = float(band_scores.sum().item())
        ranked_bands = torch.topk(band_scores, k=min(top_bands, band_scores.numel()), largest=True, sorted=True).indices
        ranked_bands = [int(index) for index in ranked_bands.tolist()]
        rank_by_band = {band_idx: rank for rank, band_idx in enumerate(ranked_bands, start=1)}
        band_indices = _selected_gaussianity_bands(num_pairs, ranked_bands, band_scope)

        q_by_band = q_complex_pairs[:, :, query_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)
        k_by_band = k_complex_pairs[:, :, key_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)
        for band_idx in band_indices:
            score = float(band_scores[band_idx].item())
            base_row = {
                "layer": layer_idx,
                "query_head": query_head_idx,
                "key_head": key_head_idx,
                "band_idx": band_idx,
                "dominant_rank": rank_by_band.get(band_idx, ""),
                "dims": f"{band_idx},{band_idx + num_pairs}",
                "score": score,
                "score_share": score / total_score if total_score > 0 else 0.0,
            }
            if inv_freq is not None:
                freq = float(inv_freq[band_idx].item())
                base_row["inv_freq"] = freq
                base_row["wavelength_tokens"] = float((2 * math.pi) / freq) if freq > 0 else float("nan")

            for tensor_name, values in (("q", q_by_band[:, band_idx]), ("k", k_by_band[:, band_idx])):
                row = dict(base_row)
                row["tensor"] = tensor_name
                row.update(summarize_complex_gaussianity(values))
                rows.append(row)
    return rows


def plot_qk_gaussianity_diagnostics(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    output_dir: Path,
    plot_top_bands: int,
    max_points: int,
    hist_bins: int,
) -> None:
    if plot_top_bands <= 0:
        return

    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "gaussianity_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for query_head_idx in selected_query_heads:
        band_indices, band_scores, key_head_idx = select_dominant_qk_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
            top_k=plot_top_bands,
        )
        total_score = float(band_scores.sum().item())

        for rank, band_idx in enumerate(band_indices, start=1):
            q_values = q_complex_pairs[:, :, query_head_idx, band_idx].reshape(-1).to(torch.complex64)
            k_values = k_complex_pairs[:, :, key_head_idx, band_idx].reshape(-1).to(torch.complex64)
            q_summary = summarize_complex_gaussianity(q_values)
            k_summary = summarize_complex_gaussianity(k_values)
            score = float(band_scores[band_idx].item())
            score_share = score / total_score if total_score > 0 else 0.0

            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14.5, 6.5), squeeze=False)
            for row_idx, (tensor_name, values, color, summary) in enumerate(
                (
                    ("Q", q_values, "tab:blue", q_summary),
                    ("K", k_values, "tab:orange", k_summary),
                )
            ):
                plot_hist_with_gaussian(
                    axes[row_idx][0],
                    values.real,
                    f"{tensor_name} real histogram",
                    color,
                    max_points=max_points,
                    bins=hist_bins,
                )
                plot_normal_qq(axes[row_idx][1], values.real, f"{tensor_name} real QQ", color, max_points=max_points)
                plot_hist_with_gaussian(
                    axes[row_idx][2],
                    values.imag,
                    f"{tensor_name} imag histogram",
                    color,
                    max_points=max_points,
                    bins=hist_bins,
                )
                plot_normal_qq(axes[row_idx][3], values.imag, f"{tensor_name} imag QQ", color, max_points=max_points)
                axes[row_idx][3].text(
                    0.98,
                    0.04,
                    f"corr={float(summary['corr_real_imag']):.2f}\n"
                    f"Mardia excess={float(summary['mardia_kurtosis_excess']):.2f}\n"
                    f"chi2 KS={float(summary['mahalanobis_chi2_ks']):.2f}",
                    transform=axes[row_idx][3].transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )

            fig.suptitle(
                f"Gaussianity diagnostics, layer {layer_idx}, query head {query_head_idx}, "
                f"key head {key_head_idx}, top{rank} band {band_idx}, C={score:.3g} ({score_share:.1%})"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(
                plot_dir / f"qk_layer{layer_idx}_qhead{query_head_idx}_kvhead{key_head_idx}_band{band_idx}_gaussianity.png",
                dpi=180,
            )
            plt.close(fig)

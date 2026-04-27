import math
from pathlib import Path
from typing import Any

import torch

from .complex_pairs import mean_resultant_length
from .dominant_bands import select_dominant_qk_bands


def load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it in your environment first.") from exc
    return plt


def maybe_sample_points(values: torch.Tensor, plot_max_points: int) -> tuple[Any, Any]:
    if values.numel() > plot_max_points:
        values = values[torch.randperm(values.numel())[:plot_max_points]]
    return values.real.cpu().numpy(), values.imag.cpu().numpy()


def compute_joint_plot_limit(value_sets: list[torch.Tensor], plot_radius_quantile: float) -> float:
    coordinates = []
    for values in value_sets:
        coordinates.append(values.real.reshape(-1).abs())
        coordinates.append(values.imag.reshape(-1).abs())
    merged = torch.cat(coordinates)
    merged = merged[torch.isfinite(merged)]
    if merged.numel() == 0:
        return 1.0

    quantile = min(max(plot_radius_quantile, 0.0), 1.0)
    quantile_limit = float(torch.quantile(merged, quantile).item())
    full_range_limit = float(merged.max().item())
    return max(quantile_limit, full_range_limit, 1e-4) * 1.08


def concentration_histogram_bins(q_values: list[float], k_values: list[float]) -> list[float]:
    values = q_values + k_values
    if not values:
        return [0.5 + 0.05 * idx for idx in range(11)]

    min_value = min(values)
    lower = 0.5 if min_value >= 0.5 else max(0.0, math.floor(min_value * 10) / 10)
    return [lower + (1.0 - lower) * idx / 10 for idx in range(11)]


def plot_qk_concentration_distribution(concentration_rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not concentration_rows:
        return

    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "concentration_distribution"
    plot_dir.mkdir(parents=True, exist_ok=True)

    q_values = [float(row["q_concentration_r"]) for row in concentration_rows]
    k_values = [float(row["k_concentration_r"]) for row in concentration_rows]
    bins = concentration_histogram_bins(q_values, k_values)
    q_weights = [100.0 / len(q_values)] * len(q_values)
    k_weights = [100.0 / len(k_values)] * len(k_values)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(
        [q_values, k_values],
        bins=bins,
        weights=[q_weights, k_weights],
        color=["tab:blue", "tab:orange"],
        alpha=0.78,
        rwidth=0.82,
        label=["Q", "K"],
    )
    ax.set_title("Concentration Distribution")
    ax.set_xlabel("Concentration R")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlim(bins[0], 1.0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(plot_dir / "qk_concentration_r_distribution.png", dpi=180)
    plt.close(fig)


def plot_qk_top_frequency_bands(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    output_dir: Path,
    plot_max_points: int,
    plot_radius_quantile: float,
    top_bands: int,
    inv_freq: torch.Tensor | None = None,
) -> None:
    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "top_frequency_bands"
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_pairs = q_complex_pairs.shape[-1]

    for query_head_idx in selected_query_heads:
        band_indices, band_scores, key_head_idx = select_dominant_qk_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
            top_k=top_bands,
        )
        total_score = float(band_scores.sum().item())
        q_by_band = q_complex_pairs[:, :, query_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)
        k_by_band = k_complex_pairs[:, :, key_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)

        ncols = min(4, len(band_indices))
        nrows = math.ceil(len(band_indices) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)

        for plot_idx, band_idx in enumerate(band_indices):
            rank = plot_idx + 1
            ax = axes[plot_idx // ncols][plot_idx % ncols]
            q_values = q_by_band[:, band_idx]
            k_values = k_by_band[:, band_idx]
            q_real, q_imag = maybe_sample_points(q_values, plot_max_points)
            k_real, k_imag = maybe_sample_points(k_values, plot_max_points)
            limit = compute_joint_plot_limit(
                value_sets=[q_values, k_values],
                plot_radius_quantile=plot_radius_quantile,
            )

            ax.scatter(q_real, q_imag, s=6, alpha=0.32, linewidths=0, color="tab:blue", label="Q")
            ax.scatter(k_real, k_imag, s=6, alpha=0.32, linewidths=0, color="tab:orange", label="K")
            ax.scatter([q_values.mean().real.item()], [q_values.mean().imag.item()], c="tab:blue", s=34, marker="x")
            ax.scatter([k_values.mean().real.item()], [k_values.mean().imag.item()], c="tab:orange", s=34, marker="x")

            score = float(band_scores[band_idx].item())
            score_share = score / total_score if total_score > 0 else 0.0
            ax.set_title(f"top{rank} f={band_idx} C={score:.3g} ({score_share:.1%})", fontsize=10)
            ax.text(
                0.98,
                0.96,
                f"E|Q|={torch.abs(q_values).mean().item():.2f}\n"
                f"E|K|={torch.abs(k_values).mean().item():.2f}\n"
                f"RQ={mean_resultant_length(q_values):.2f}\n"
                f"RK={mean_resultant_length(k_values):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            if inv_freq is not None:
                freq = float(inv_freq[band_idx].item())
                if freq > 0:
                    ax.text(
                        0.02,
                        0.04,
                        f"lambda={(2 * math.pi) / freq:.1f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=8,
                    )
            if plot_idx == 0:
                ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.grid(alpha=0.14)
            ax.tick_params(labelsize=8)

        for empty_idx in range(len(band_indices), nrows * ncols):
            axes[empty_idx // ncols][empty_idx % ncols].axis("off")

        fig.suptitle(
            f"Top pre-RoPE Q/K frequency bands, layer {layer_idx}, "
            f"query head {query_head_idx}, key head {key_head_idx}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(
            plot_dir / f"qk_layer{layer_idx}_qhead{query_head_idx}_kvhead{key_head_idx}_top_bands.png",
            dpi=180,
        )
        plt.close(fig)


def plot_qk_top1_heads_by_layer(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    output_dir: Path,
    plot_max_points: int,
    plot_radius_quantile: float,
    inv_freq: torch.Tensor | None = None,
) -> None:
    if not selected_query_heads:
        return

    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "top1_heads_by_layer"
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_pairs = q_complex_pairs.shape[-1]
    ncols = min(8, len(selected_query_heads))
    nrows = math.ceil(len(selected_query_heads) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.55 * ncols, 3.5 * nrows), squeeze=False)

    for plot_idx, query_head_idx in enumerate(selected_query_heads):
        ax = axes[plot_idx // ncols][plot_idx % ncols]
        band_indices, band_scores, key_head_idx = select_dominant_qk_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
            top_k=1,
        )
        band_idx = band_indices[0]
        total_score = float(band_scores.sum().item())
        score = float(band_scores[band_idx].item())
        score_share = score / total_score if total_score > 0 else 0.0

        q_values = q_complex_pairs[:, :, query_head_idx, band_idx].reshape(-1).to(torch.complex64)
        k_values = k_complex_pairs[:, :, key_head_idx, band_idx].reshape(-1).to(torch.complex64)
        q_real, q_imag = maybe_sample_points(q_values, plot_max_points)
        k_real, k_imag = maybe_sample_points(k_values, plot_max_points)
        limit = compute_joint_plot_limit(
            value_sets=[q_values, k_values],
            plot_radius_quantile=plot_radius_quantile,
        )

        ax.scatter(q_real, q_imag, s=4, alpha=0.26, linewidths=0, color="tab:blue", label="Q")
        ax.scatter(k_real, k_imag, s=4, alpha=0.26, linewidths=0, color="tab:orange", label="K")
        ax.scatter([q_values.mean().real.item()], [q_values.mean().imag.item()], c="tab:blue", s=24, marker="x")
        ax.scatter([k_values.mean().real.item()], [k_values.mean().imag.item()], c="tab:orange", s=24, marker="x")

        ax.set_title(
            f"qh={query_head_idx} kh={key_head_idx} f={band_idx}\nC={score:.2g} ({score_share:.1%})",
            fontsize=8,
        )
        ax.text(
            0.98,
            0.96,
            f"RQ={mean_resultant_length(q_values):.2f}\nRK={mean_resultant_length(k_values):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        if inv_freq is not None:
            freq = float(inv_freq[band_idx].item())
            if freq > 0:
                ax.text(
                    0.02,
                    0.04,
                    f"lambda={(2 * math.pi) / freq:.1f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                )
        if plot_idx == 0:
            ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=7)

    for empty_idx in range(len(selected_query_heads), nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.suptitle(f"Top-1 pre-RoPE Q/K dominant band per query head, layer {layer_idx}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_dir / f"qk_layer{layer_idx}_top1_heads.png", dpi=180)
    plt.close(fig)


def plot_qk_frequency_grids(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    output_dir: Path,
    plot_max_points: int,
    plot_radius_quantile: float,
    top_bands: int,
    inv_freq: torch.Tensor | None = None,
) -> None:
    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "frequency_grids"
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_pairs = q_complex_pairs.shape[-1]
    ncols = min(8, num_pairs)
    nrows = math.ceil(num_pairs / ncols)

    for query_head_idx in selected_query_heads:
        band_indices, band_scores, key_head_idx = select_dominant_qk_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
            top_k=top_bands,
        )
        dominant_rank = {band_idx: rank for rank, band_idx in enumerate(band_indices, start=1)}

        q_by_band = q_complex_pairs[:, :, query_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)
        k_by_band = k_complex_pairs[:, :, key_head_idx, :].reshape(-1, num_pairs).to(torch.complex64)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.3 * ncols, 3.25 * nrows), squeeze=False)
        for band_idx in range(num_pairs):
            ax = axes[band_idx // ncols][band_idx % ncols]
            q_values = q_by_band[:, band_idx]
            k_values = k_by_band[:, band_idx]
            q_real, q_imag = maybe_sample_points(q_values, plot_max_points)
            k_real, k_imag = maybe_sample_points(k_values, plot_max_points)
            limit = compute_joint_plot_limit(
                value_sets=[q_values, k_values],
                plot_radius_quantile=plot_radius_quantile,
            )

            ax.scatter(q_real, q_imag, s=4, alpha=0.28, linewidths=0, color="tab:blue")
            ax.scatter(k_real, k_imag, s=4, alpha=0.28, linewidths=0, color="tab:orange")
            ax.scatter([q_values.mean().real.item()], [q_values.mean().imag.item()], c="tab:blue", s=24, marker="x")
            ax.scatter([k_values.mean().real.item()], [k_values.mean().imag.item()], c="tab:orange", s=24, marker="x")

            title = f"f={band_idx} C={band_scores[band_idx].item():.2g}"
            if band_idx in dominant_rank:
                title = f"top{dominant_rank[band_idx]} " + title
                for spine in ax.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(1.6)
            ax.set_title(title, fontsize=8)
            ax.text(
                0.98,
                0.96,
                f"RQ={mean_resultant_length(q_values):.2f}\nRK={mean_resultant_length(k_values):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            if inv_freq is not None and band_idx in dominant_rank:
                freq = float(inv_freq[band_idx].item())
                if freq > 0:
                    ax.text(
                        0.02,
                        0.04,
                        f"lambda={(2 * math.pi) / freq:.1f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                    )
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.grid(alpha=0.12)
            ax.tick_params(labelsize=7)

        for empty_idx in range(num_pairs, nrows * ncols):
            axes[empty_idx // ncols][empty_idx % ncols].axis("off")

        fig.suptitle(
            f"Pre-RoPE Q/K by frequency band, layer {layer_idx}, "
            f"query head {query_head_idx}, key head {key_head_idx}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            plot_dir / f"qk_layer{layer_idx}_qhead{query_head_idx}_kvhead{key_head_idx}_frequency_grid.png",
            dpi=180,
        )
        plt.close(fig)

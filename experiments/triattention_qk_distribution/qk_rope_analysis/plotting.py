import math
from pathlib import Path
from typing import Any

import torch

from .analysis import PAIR_DEFINITION, flatten_head_metric_rows, summarize_layer_metric_rows


SUMMARY_METRIC_SPECS = [
    ("top1_band_share", "top1 band share", "viridis", 0.0, 1.0),
    ("top4_band_share", "top4 band share", "viridis", 0.0, 1.0),
    ("band_entropy", "normalized band entropy", "magma", 0.0, 1.0),
    ("dominant_pair_idx", "dominant pair idx", "turbo", 0.0, None),
    ("vector_l2_mean", "mean token L2 norm", "cividis", 0.0, None),
    ("axis_ratio", "complex cloud axis ratio", "plasma", 1.0, None),
]


def load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it in your environment first.") from exc
    return plt


def get_tick_positions(size: int, max_ticks: int = 12) -> tuple[list[int], list[str]]:
    if size <= 0:
        return [], []
    step = max(1, math.ceil(size / max_ticks))
    positions = list(range(0, size, step))
    if positions[-1] != size - 1:
        positions.append(size - 1)
    return positions, [str(pos) for pos in positions]


def plot_metric_heatmap(
    plt,
    matrix: torch.Tensor,
    title: str,
    output_path: Path,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    y_tick_labels: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix.cpu().numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x_positions, x_labels = get_tick_positions(matrix.shape[1])
    ax.set_xticks(x_positions, labels=x_labels)

    if y_tick_labels is None:
        y_positions, y_labels = get_tick_positions(matrix.shape[0])
        ax.set_yticks(y_positions, labels=y_labels)
    else:
        ax.set_yticks(list(range(len(y_tick_labels))), labels=y_tick_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_metric_matrix(
    head_rows: list[dict[str, Any]],
    layers: list[int],
    heads: list[int],
    metric_key: str,
) -> torch.Tensor:
    layer_to_row = {layer_idx: row_idx for row_idx, layer_idx in enumerate(layers)}
    head_to_col = {head_idx: col_idx for col_idx, head_idx in enumerate(heads)}
    matrix = torch.full((len(layers), len(heads)), float("nan"), dtype=torch.float32)
    for row in head_rows:
        matrix[layer_to_row[row["layer"]], head_to_col[row["head"]]] = float(row[metric_key])
    return matrix


def plot_summary_metric_heatmaps(
    plt,
    head_rows: list[dict[str, Any]],
    tensor_name: str,
    plot_dir: Path,
) -> None:
    layers = sorted({row["layer"] for row in head_rows})
    heads = sorted({row["head"] for row in head_rows})

    for metric_key, metric_label, cmap, vmin, vmax in SUMMARY_METRIC_SPECS:
        matrix = build_metric_matrix(head_rows=head_rows, layers=layers, heads=heads, metric_key=metric_key)
        plot_metric_heatmap(
            plt=plt,
            matrix=matrix,
            title=f"{tensor_name.upper()} layer-head {metric_label}",
            output_path=plot_dir / f"{tensor_name}_{metric_key}_heatmap.png",
            x_label="Head index",
            y_label="Layer index",
            colorbar_label=metric_label,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )


def plot_layer_trends(
    plt,
    layer_rows: list[dict[str, Any]],
    tensor_name: str,
    plot_dir: Path,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
    layer_indices = [row["layer"] for row in layer_rows]

    axes[0].plot(layer_indices, [row["mean_top1_band_share"] for row in layer_rows], marker="o")
    axes[0].plot(layer_indices, [row["max_top1_band_share"] for row in layer_rows], marker="x")
    axes[0].set_title(f"{tensor_name.upper()} band concentration by layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Energy share")
    axes[0].legend(["mean top1", "max top1"])
    axes[0].grid(alpha=0.2)

    axes[1].plot(layer_indices, [row["mean_band_entropy"] for row in layer_rows], marker="o")
    axes[1].plot(layer_indices, [row["mean_effective_num_bands"] for row in layer_rows], marker="x")
    axes[1].set_title(f"{tensor_name.upper()} entropy / effective bands")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Value")
    axes[1].legend(["mean entropy", "effective bands"])
    axes[1].grid(alpha=0.2)

    axes[2].plot(layer_indices, [row["mean_vector_l2"] for row in layer_rows], marker="o")
    axes[2].plot(layer_indices, [row["mean_axis_ratio"] for row in layer_rows], marker="x")
    axes[2].set_title(f"{tensor_name.upper()} magnitude / anisotropy")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Value")
    axes[2].legend(["mean vector L2", "mean axis ratio"])
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_dir / f"{tensor_name}_layer_trends.png", dpi=180)
    plt.close(fig)


def plot_layer_pair_energy(
    plt,
    tensor_summary: dict[str, Any],
    tensor_name: str,
    plot_dir: Path,
) -> None:
    for layer_key in sorted(tensor_summary.keys(), key=int):
        layer_summary = tensor_summary[layer_key]
        head_labels = sorted(layer_summary["per_head"].keys(), key=int)
        num_pairs = int(layer_summary["num_pairs"])
        energy_matrix = torch.zeros((len(head_labels), num_pairs), dtype=torch.float32)
        for row_idx, head_key in enumerate(head_labels):
            per_pair = layer_summary["per_head"][head_key]["frequency_band_distribution"]["per_pair"]
            for pair_summary in per_pair:
                energy_matrix[row_idx, pair_summary["pair_idx"]] = float(pair_summary["energy_share"])

        plot_metric_heatmap(
            plt=plt,
            matrix=energy_matrix,
            title=f"{tensor_name.upper()} layer {layer_key} head-pair energy share",
            output_path=plot_dir / f"{tensor_name}_layer{layer_key}_head_pair_energy.png",
            x_label="RoPE pair index",
            y_label="Head index",
            colorbar_label="energy share",
            cmap="magma",
            vmin=0.0,
            vmax=float(energy_matrix.max().item()) if energy_matrix.numel() else 1.0,
            y_tick_labels=head_labels,
        )


def maybe_plot_summary(
    summary: dict[str, Any],
    tensor_names: list[str],
    output_dir: Path,
) -> None:
    plt = load_matplotlib_pyplot()
    plot_dir = output_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for tensor_name in tensor_names:
        tensor_summary = summary[tensor_name]
        head_rows = flatten_head_metric_rows(tensor_summary, tensor_name=tensor_name)
        if not head_rows:
            continue

        plot_summary_metric_heatmaps(plt=plt, head_rows=head_rows, tensor_name=tensor_name, plot_dir=plot_dir)
        plot_layer_trends(
            plt=plt,
            layer_rows=summarize_layer_metric_rows(head_rows),
            tensor_name=tensor_name,
            plot_dir=plot_dir,
        )
        plot_layer_pair_energy(plt=plt, tensor_summary=tensor_summary, tensor_name=tensor_name, plot_dir=plot_dir)


def compute_plot_limit(head_values: torch.Tensor, plot_radius_quantile: float) -> float:
    real_all = head_values.real.reshape(-1).abs()
    imag_all = head_values.imag.reshape(-1).abs()
    limit_tensor = torch.cat([real_all, imag_all])
    limit = float(torch.quantile(limit_tensor, plot_radius_quantile).item())
    return max(limit, 1e-4)


def maybe_sample_points(values: torch.Tensor, plot_max_points: int) -> tuple[Any, Any]:
    if values.numel() > plot_max_points:
        perm = torch.randperm(values.numel())[:plot_max_points]
        values = values[perm]
    return values.real.cpu().numpy(), values.imag.cpu().numpy()


def maybe_plot_complex_pairs(
    complex_pairs: torch.Tensor,
    tensor_name: str,
    layer_idx: int,
    selected_heads: list[int],
    output_dir: Path,
    plot_type: str,
    plot_bins: int,
    plot_max_points: int,
    plot_radius_quantile: float,
) -> None:
    plt = load_matplotlib_pyplot()

    num_pairs = complex_pairs.shape[-1]
    ncols = min(4, num_pairs)
    nrows = math.ceil(num_pairs / ncols)

    for head_idx in selected_heads:
        head_values = complex_pairs[:, :, head_idx, :].reshape(-1, num_pairs)
        limit = compute_plot_limit(head_values=head_values, plot_radius_quantile=plot_radius_quantile)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for pair_idx in range(num_pairs):
            ax = axes[pair_idx // ncols][pair_idx % ncols]
            values = head_values[:, pair_idx]

            if plot_type == "hist2d":
                ax.hist2d(
                    values.real.cpu().numpy(),
                    values.imag.cpu().numpy(),
                    bins=plot_bins,
                    range=[[-limit, limit], [-limit, limit]],
                    cmap="magma",
                )
            else:
                real, imag = maybe_sample_points(values=values, plot_max_points=plot_max_points)
                ax.scatter(real, imag, s=5, alpha=0.35, linewidths=0)

            center = values.mean()
            ax.scatter([center.real.item()], [center.imag.item()], c="cyan", s=25, marker="x")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.set_title(f"pair {pair_idx} dims ({pair_idx}, {pair_idx + num_pairs})")
            ax.grid(alpha=0.15)

        for empty_idx in range(num_pairs, nrows * ncols):
            axes[empty_idx // ncols][empty_idx % ncols].axis("off")

        fig.suptitle(
            f"{tensor_name.upper()} layer {layer_idx} head {head_idx} pre-RoPE complex pairs\n"
            f"actual Qwen3 pairing: (i, i + head_dim/2)"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{tensor_name}_layer{layer_idx}_head{head_idx}_pair_grid.png", dpi=180)
        plt.close(fig)

        centroid_fig, centroid_ax = plt.subplots(figsize=(6, 6))
        centroids = head_values.mean(dim=0)
        centroid_ax.scatter(centroids.real.cpu().numpy(), centroids.imag.cpu().numpy(), s=30)
        for pair_idx, centroid in enumerate(centroids):
            centroid_ax.text(centroid.real.item(), centroid.imag.item(), str(pair_idx), fontsize=8)
        centroid_ax.set_xlim(-limit, limit)
        centroid_ax.set_ylim(-limit, limit)
        centroid_ax.set_aspect("equal")
        centroid_ax.grid(alpha=0.2)
        centroid_ax.set_title(
            f"{tensor_name.upper()} layer {layer_idx} head {head_idx} pair centroids\n"
            f"{PAIR_DEFINITION}"
        )
        centroid_fig.tight_layout()
        centroid_fig.savefig(output_dir / f"{tensor_name}_layer{layer_idx}_head{head_idx}_centroids.png", dpi=180)
        plt.close(centroid_fig)

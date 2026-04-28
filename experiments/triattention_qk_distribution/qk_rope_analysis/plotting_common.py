from typing import Any

import torch


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


def maybe_subsample_axis(values: torch.Tensor, dim: int, max_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    axis_size = values.shape[dim]
    if max_items <= 0 or axis_size <= max_items:
        return values, torch.arange(axis_size, device=values.device)

    indices = torch.linspace(0, axis_size - 1, steps=max_items, device=values.device).round().to(dtype=torch.long)
    return values.index_select(dim, indices), indices


def robust_nonnegative_vmax(values: torch.Tensor, color_quantile: float) -> float:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return 1.0

    quantile = min(max(color_quantile, 0.0), 1.0)
    if quantile <= 0.0:
        vmax = float(finite_values.max().item())
    else:
        vmax = float(torch.quantile(finite_values, quantile).item())
    return max(vmax, 1e-6)

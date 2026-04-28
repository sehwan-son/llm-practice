"""Compatibility re-exports for plotting helpers.

New code should import from the focused modules directly:
`plotting_common`, `key_magnitude_plots`, or `qk_cloud_plots`.
"""

from .key_magnitude_plots import (
    plot_pre_rope_key_magnitude_heatmap,
    plot_pre_rope_key_magnitude_plots,
    plot_pre_rope_key_magnitude_surface3d,
    select_readable_surface_channels,
)
from .plotting_common import (
    compute_joint_plot_limit,
    load_matplotlib_pyplot,
    maybe_sample_points,
    maybe_subsample_axis,
    robust_nonnegative_vmax,
)
from .qk_cloud_plots import (
    concentration_histogram_bins,
    plot_qk_concentration_distribution,
    plot_qk_frequency_grids,
    plot_qk_top1_heads_by_layer,
    plot_qk_top_frequency_bands,
)

__all__ = [
    "compute_joint_plot_limit",
    "concentration_histogram_bins",
    "load_matplotlib_pyplot",
    "maybe_sample_points",
    "maybe_subsample_axis",
    "plot_pre_rope_key_magnitude_heatmap",
    "plot_pre_rope_key_magnitude_plots",
    "plot_pre_rope_key_magnitude_surface3d",
    "plot_qk_concentration_distribution",
    "plot_qk_frequency_grids",
    "plot_qk_top1_heads_by_layer",
    "plot_qk_top_frequency_bands",
    "robust_nonnegative_vmax",
    "select_readable_surface_channels",
]

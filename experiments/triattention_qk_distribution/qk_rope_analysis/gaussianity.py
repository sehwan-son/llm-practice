"""Compatibility re-exports for Gaussianity helpers.

New code should import from `normality`, `band_gaussianity`, or
`centered_dimension_gaussianity` directly.
"""

from .band_gaussianity import build_qk_gaussianity_rows, plot_qk_gaussianity_diagnostics
from .centered_dimension_gaussianity import (
    build_centered_dimension_gaussianity_rows,
    mean_center_pre_rope_vectors,
    plot_centered_dimension_gaussianity_diagnostics,
)
from .normality import summarize_complex_gaussianity, summarize_univariate_normality


__all__ = [
    "build_centered_dimension_gaussianity_rows",
    "build_qk_gaussianity_rows",
    "mean_center_pre_rope_vectors",
    "plot_centered_dimension_gaussianity_diagnostics",
    "plot_qk_gaussianity_diagnostics",
    "summarize_complex_gaussianity",
    "summarize_univariate_normality",
]

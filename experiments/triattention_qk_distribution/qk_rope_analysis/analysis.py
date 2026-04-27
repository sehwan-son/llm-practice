"""Backward-compatible facade for the qk_rope_analysis package.

New code should prefer the focused modules in this package. This file keeps
older imports such as ``qk_rope_analysis.analysis.write_csv`` working.
"""

from .complex_pairs import flatten_complex_cloud, mean_resultant_length, to_rope_complex_pairs
from .config import (
    build_prompt_text,
    default_output_dir,
    parse_index_selection,
    resolve_device,
    resolve_dtype,
    resolve_tensor_names,
    sanitize_model_name,
)
from .constants import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    EXPERIMENT_ROOT,
    PAIR_DEFINITION,
    PAIRING_MODE,
    PAIRING_NOTE,
)
from .dominant_bands import (
    build_qk_dominant_band_rows,
    map_query_head_to_key_head,
    qk_band_contribution_scores,
    select_dominant_qk_bands,
)
from .metrics import build_combined_metric_rows, flatten_head_metric_rows, summarize_layer_metric_rows
from .modeling import capture_pre_rope_qk, get_decoder_layers, get_layer_rope_inv_freq, load_model, load_tokenizer
from .serialization import write_csv, write_json
from .summaries import (
    summarize_complex_cloud,
    summarize_complex_pairs,
    summarize_frequency_band_distribution,
    summarize_head_value_distribution,
    summarize_real_values,
)


__all__ = [
    "DEFAULT_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "EXPERIMENT_ROOT",
    "PAIR_DEFINITION",
    "PAIRING_MODE",
    "PAIRING_NOTE",
    "build_combined_metric_rows",
    "build_prompt_text",
    "build_qk_dominant_band_rows",
    "capture_pre_rope_qk",
    "default_output_dir",
    "flatten_complex_cloud",
    "flatten_head_metric_rows",
    "get_decoder_layers",
    "get_layer_rope_inv_freq",
    "load_model",
    "load_tokenizer",
    "map_query_head_to_key_head",
    "mean_resultant_length",
    "parse_index_selection",
    "qk_band_contribution_scores",
    "resolve_device",
    "resolve_dtype",
    "resolve_tensor_names",
    "sanitize_model_name",
    "select_dominant_qk_bands",
    "summarize_complex_cloud",
    "summarize_complex_pairs",
    "summarize_frequency_band_distribution",
    "summarize_head_value_distribution",
    "summarize_layer_metric_rows",
    "summarize_real_values",
    "to_rope_complex_pairs",
    "write_csv",
    "write_json",
]

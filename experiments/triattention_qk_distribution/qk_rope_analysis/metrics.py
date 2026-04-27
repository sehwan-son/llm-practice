from typing import Any


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

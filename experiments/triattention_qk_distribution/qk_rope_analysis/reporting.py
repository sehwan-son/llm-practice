from pathlib import Path
from typing import Any

from .constants import PAIRING_NOTE


def format_dominant_bands(head_summary: dict[str, Any]) -> str:
    band_dist = head_summary["frequency_band_distribution"]
    dominant_bands = []
    for pair_summary in band_dist["dominant_pairs"][:3]:
        band_label = f"pair {pair_summary['pair_idx']} ({pair_summary['energy_share'] * 100:.1f}%"
        if "wavelength_tokens" in pair_summary:
            band_label += f", lambda={pair_summary['wavelength_tokens']:.1f}"
        dominant_bands.append(f"{band_label})")
    return ", ".join(dominant_bands)


def print_analysis_report(output_dir: Path, tensor_names: list[str], summary: dict[str, Any]) -> None:
    print(f"Saved artifacts to: {output_dir}")
    print(f"RoPE pairing note: {PAIRING_NOTE}")

    for tensor_name in tensor_names:
        print(f"\n[{tensor_name.upper()}] pre-RoPE head distribution")
        for layer_key, layer_summary in summary[tensor_name].items():
            for head_key, head_summary in layer_summary["per_head"].items():
                aggregate = head_summary["aggregate"]
                raw_dist = head_summary["raw_head_distribution"]
                band_dist = head_summary["frequency_band_distribution"]
                print(
                    f"  layer {layer_key} head {head_key}: "
                    f"value_std={raw_dist['value']['std']:.4f}, "
                    f"vector_l2_mean={raw_dist['vector_l2_norm']['mean']:.4f}, "
                    f"top1_band_share={band_dist['top1_energy_share']:.4f}, "
                    f"top4_band_share={band_dist['top4_energy_share']:.4f}, "
                    f"band_entropy={band_dist['normalized_entropy']:.4f}, "
                    f"center_radius={aggregate['center_radius']:.4f}, "
                    f"mean_radius={aggregate['mean_radius']:.4f}, "
                    f"major_axis_std={aggregate['major_axis_std']:.4f}, "
                    f"minor_axis_std={aggregate['minor_axis_std']:.4f}, "
                    f"axis_ratio={aggregate['axis_ratio']:.4f}, "
                    f"dominant_bands=[{format_dominant_bands(head_summary)}]"
                )

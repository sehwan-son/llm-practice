from typing import Any

import torch

from .complex_pairs import mean_resultant_length


def map_query_head_to_key_head(query_head_idx: int, num_query_heads: int, num_key_heads: int) -> int:
    if num_query_heads == num_key_heads:
        return query_head_idx
    if num_query_heads % num_key_heads != 0:
        raise ValueError(
            f"Cannot map {num_query_heads} query heads to {num_key_heads} key heads with an integer GQA ratio."
        )
    return query_head_idx // (num_query_heads // num_key_heads)


def qk_band_contribution_scores(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    query_head_idx: int,
) -> tuple[torch.Tensor, int]:
    if q_complex_pairs.shape[:2] != k_complex_pairs.shape[:2]:
        raise ValueError(
            "Q and K complex tensors must share batch/sequence shape, "
            f"got {tuple(q_complex_pairs.shape[:2])} and {tuple(k_complex_pairs.shape[:2])}."
        )
    if q_complex_pairs.shape[-1] != k_complex_pairs.shape[-1]:
        raise ValueError(
            f"Q and K must have the same number of RoPE pairs, got {q_complex_pairs.shape[-1]} "
            f"and {k_complex_pairs.shape[-1]}."
        )

    key_head_idx = map_query_head_to_key_head(query_head_idx, q_complex_pairs.shape[2], k_complex_pairs.shape[2])
    q_values = q_complex_pairs[:, :, query_head_idx, :].to(torch.complex64)
    k_values = k_complex_pairs[:, :, key_head_idx, :].to(torch.complex64)
    scores = torch.abs(q_values).mean(dim=(0, 1)) * torch.abs(k_values).mean(dim=(0, 1))
    return scores.to(torch.float32), key_head_idx


def select_dominant_qk_bands(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    query_head_idx: int,
    top_k: int,
) -> tuple[list[int], torch.Tensor, int]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    scores, key_head_idx = qk_band_contribution_scores(q_complex_pairs, k_complex_pairs, query_head_idx)
    _, indices = torch.topk(scores, k=min(top_k, scores.numel()), largest=True, sorted=True)
    return indices.tolist(), scores, key_head_idx


def build_qk_dominant_band_rows(
    q_complex_pairs: torch.Tensor,
    k_complex_pairs: torch.Tensor,
    layer_idx: int,
    selected_query_heads: list[int],
    top_bands: int,
    inv_freq: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for query_head_idx in selected_query_heads:
        band_indices, band_scores, key_head_idx = select_dominant_qk_bands(
            q_complex_pairs=q_complex_pairs,
            k_complex_pairs=k_complex_pairs,
            query_head_idx=query_head_idx,
            top_k=top_bands,
        )
        total_score = float(band_scores.sum().item())
        for rank, band_idx in enumerate(band_indices, start=1):
            q_values = q_complex_pairs[:, :, query_head_idx, band_idx].reshape(-1).to(torch.complex64)
            k_values = k_complex_pairs[:, :, key_head_idx, band_idx].reshape(-1).to(torch.complex64)
            q_center = q_values.mean()
            k_center = k_values.mean()
            score = float(band_scores[band_idx].item())
            row = {
                "layer": layer_idx,
                "query_head": query_head_idx,
                "key_head": key_head_idx,
                "rank": rank,
                "band_idx": band_idx,
                "dims": f"{band_idx},{band_idx + q_complex_pairs.shape[-1]}",
                "score": score,
                "score_share": score / total_score if total_score > 0 else 0.0,
                "q_expected_norm": float(torch.abs(q_values).mean().item()),
                "k_expected_norm": float(torch.abs(k_values).mean().item()),
                "q_center_real": float(q_center.real.item()),
                "q_center_imag": float(q_center.imag.item()),
                "q_center_abs": float(torch.abs(q_center).item()),
                "k_center_real": float(k_center.real.item()),
                "k_center_imag": float(k_center.imag.item()),
                "k_center_abs": float(torch.abs(k_center).item()),
                "q_mean_resultant_length": mean_resultant_length(q_values),
                "k_mean_resultant_length": mean_resultant_length(k_values),
            }
            if inv_freq is not None:
                freq = float(inv_freq[band_idx].item())
                row["inv_freq"] = freq
                row["wavelength_tokens"] = float((2 * torch.pi) / inv_freq[band_idx].item()) if freq > 0 else float("nan")
            rows.append(row)
    return rows

import torch


def to_rope_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [batch, seq, heads, head_dim], got {tuple(tensor.shape)}.")
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    half = head_dim // 2
    real = tensor[..., :half]
    imag = tensor[..., half:]
    return torch.complex(real, imag)


def mean_resultant_length(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("Cannot compute mean resultant length for an empty complex cloud.")

    values = values.to(torch.complex64).reshape(-1)
    mean_radius = torch.abs(values).mean()
    if mean_radius.item() <= 0:
        return 0.0
    return float((torch.abs(values.mean()) / mean_radius).item())

from pathlib import Path

import torch

from .plotting_common import load_matplotlib_pyplot, maybe_subsample_axis, robust_nonnegative_vmax


def select_readable_surface_channels(magnitude: torch.Tensor, max_channels: int) -> torch.Tensor:
    total_channels = magnitude.shape[1]
    if max_channels <= 0 or total_channels <= max_channels:
        return torch.arange(total_channels, device=magnitude.device)

    uniform_count = max(1, int(max_channels * 0.65))
    salient_count = max_channels - uniform_count
    uniform_indices = torch.linspace(
        0,
        total_channels - 1,
        steps=uniform_count,
        device=magnitude.device,
    ).round().to(dtype=torch.long)

    if salient_count <= 0:
        return torch.unique(uniform_indices, sorted=True)

    channel_scores = magnitude.mean(dim=0)
    top_indices = torch.topk(channel_scores, k=min(salient_count, total_channels)).indices
    return torch.unique(torch.cat([uniform_indices, top_indices]), sorted=True)


def plot_pre_rope_key_magnitude_heatmap(
    plt,
    magnitude: torch.Tensor,
    layer_idx: int,
    batch_idx: int,
    batch_size: int,
    output_dir: Path,
    max_tokens: int,
    max_channels: int,
    color_quantile: float,
) -> None:
    plot_dir = output_dir / "pre_rope_key_magnitude"
    plot_dir.mkdir(parents=True, exist_ok=True)

    magnitude, token_indices = maybe_subsample_axis(magnitude, dim=0, max_items=max_tokens)
    magnitude, channel_indices = maybe_subsample_axis(magnitude, dim=1, max_items=max_channels)
    vmax = robust_nonnegative_vmax(magnitude, color_quantile=color_quantile)
    display = torch.nan_to_num(magnitude, nan=0.0, posinf=vmax, neginf=0.0)
    channel_start = int(channel_indices[0].item()) if channel_indices.numel() else 0
    channel_end = int(channel_indices[-1].item()) + 1 if channel_indices.numel() else magnitude.shape[1]
    token_start = int(token_indices[0].item()) if token_indices.numel() else 0
    token_end = int(token_indices[-1].item()) + 1 if token_indices.numel() else magnitude.shape[0]

    fig_width = max(6.0, min(14.0, display.shape[1] / 360))
    fig_height = max(4.0, min(9.0, display.shape[0] / 320))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        display.cpu().numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        extent=(channel_start, channel_end, token_end, token_start),
    )
    ax.set_title(f"Layer {layer_idx} Keys (pre-RoPE)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Token")
    colorbar_extend = "max" if float(display.max().item()) > vmax else "neither"
    colorbar = fig.colorbar(image, ax=ax, pad=0.015, extend=colorbar_extend)
    colorbar.set_label("Magnitude")
    fig.tight_layout()

    suffix = "" if batch_size == 1 else f"_batch{batch_idx}"
    fig.savefig(plot_dir / f"k_layer{layer_idx}{suffix}_pre_rope_key_magnitude.png", dpi=180)
    plt.close(fig)


def plot_pre_rope_key_magnitude_surface3d(
    plt,
    magnitude: torch.Tensor,
    layer_idx: int,
    batch_idx: int,
    batch_size: int,
    output_dir: Path,
    max_tokens: int,
    max_channels: int,
    color_quantile: float,
    elev: float,
    azim: float,
) -> None:
    from matplotlib.ticker import MaxNLocator

    plot_dir = output_dir / "pre_rope_key_magnitude_3d"
    plot_dir.mkdir(parents=True, exist_ok=True)

    magnitude, token_indices = maybe_subsample_axis(magnitude, dim=0, max_items=max_tokens)
    channel_indices = select_readable_surface_channels(magnitude, max_channels=max_channels)
    magnitude = magnitude.index_select(dim=1, index=channel_indices)
    vmax = robust_nonnegative_vmax(magnitude, color_quantile=color_quantile)
    display = torch.nan_to_num(magnitude, nan=0.0, posinf=vmax, neginf=0.0).clamp(min=0.0, max=vmax)

    token_grid, channel_grid = torch.meshgrid(token_indices, channel_indices, indexing="ij")
    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        channel_grid.cpu().numpy(),
        token_grid.cpu().numpy(),
        display.cpu().numpy(),
        cmap="viridis",
        linewidth=0,
        antialiased=False,
        shade=True,
        vmin=0.0,
        vmax=vmax,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Layer {layer_idx} Keys (pre-RoPE)", pad=16)
    ax.set_xlabel("Channel", labelpad=10)
    ax.set_ylabel("Token", labelpad=10)
    ax.set_zlabel("Magnitude", labelpad=8)
    ax.set_zlim(0.0, vmax)
    ax.set_box_aspect((2.3, 1.35, 0.62))
    ax.tick_params(axis="both", labelsize=8, pad=2)
    ax.tick_params(axis="z", labelsize=8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(alpha=0.18)
    colorbar_extend = "max" if float(magnitude.max().item()) > vmax else "neither"
    colorbar = fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, extend=colorbar_extend)
    colorbar.set_label(f"Magnitude, clipped at q={color_quantile:g}")
    fig.tight_layout()

    suffix = "" if batch_size == 1 else f"_batch{batch_idx}"
    fig.savefig(plot_dir / f"k_layer{layer_idx}{suffix}_pre_rope_key_magnitude_3d.png", dpi=190)
    plt.close(fig)


def plot_pre_rope_key_magnitude_plots(
    k_tensor: torch.Tensor,
    layer_idx: int,
    output_dir: Path,
    max_tokens: int,
    max_channels: int,
    color_quantile: float,
    plot_kind: str,
    surface_max_tokens: int,
    surface_max_channels: int,
    surface_elev: float,
    surface_azim: float,
) -> None:
    if k_tensor.ndim != 4:
        raise ValueError(f"Expected pre-RoPE K tensor [batch, seq, heads, head_dim], got {tuple(k_tensor.shape)}.")

    plt = load_matplotlib_pyplot()
    batch_size, seq_len, num_key_heads, head_dim = k_tensor.shape
    total_channels = num_key_heads * head_dim

    for batch_idx in range(batch_size):
        magnitude = k_tensor[batch_idx].abs().reshape(seq_len, total_channels)
        if plot_kind in {"heatmap", "both"}:
            plot_pre_rope_key_magnitude_heatmap(
                plt=plt,
                magnitude=magnitude,
                layer_idx=layer_idx,
                batch_idx=batch_idx,
                batch_size=batch_size,
                output_dir=output_dir,
                max_tokens=max_tokens,
                max_channels=max_channels,
                color_quantile=color_quantile,
            )
        if plot_kind in {"surface3d", "both"}:
            plot_pre_rope_key_magnitude_surface3d(
                plt=plt,
                magnitude=magnitude,
                layer_idx=layer_idx,
                batch_idx=batch_idx,
                batch_size=batch_size,
                output_dir=output_dir,
                max_tokens=surface_max_tokens,
                max_channels=surface_max_channels,
                color_quantile=color_quantile,
                elev=surface_elev,
                azim=surface_azim,
            )

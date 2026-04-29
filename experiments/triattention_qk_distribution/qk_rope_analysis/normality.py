import math
from statistics import NormalDist

import torch


NORMAL = NormalDist()
EPS = 1e-12


def finite_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1).to(dtype=torch.float64)
    return values[torch.isfinite(values)]


def format_pvalue(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def jarque_bera_pvalue(jarque_bera: float) -> float:
    if not math.isfinite(jarque_bera):
        return float("nan")
    return math.exp(-0.5 * max(jarque_bera, 0.0))


def summarize_univariate_normality(values: torch.Tensor) -> dict[str, float | int]:
    values = finite_1d(values)
    n = int(values.numel())
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "excess_kurtosis": float("nan"),
            "jarque_bera": float("nan"),
            "jarque_bera_pvalue": float("nan"),
        }

    mean = values.mean()
    centered = values - mean
    variance = (centered * centered).mean()
    std = torch.sqrt(variance)
    if float(std.item()) <= EPS:
        skewness = float("nan")
        excess_kurtosis = float("nan")
        jarque_bera = float("nan")
    else:
        z = centered / std
        skewness = float((z**3).mean().item())
        excess_kurtosis = float((z**4).mean().item() - 3.0)
        jarque_bera = float((n / 6.0) * (skewness * skewness + 0.25 * excess_kurtosis * excess_kurtosis))

    return {
        "n": n,
        "mean": float(mean.item()),
        "std": float(std.item()),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera": jarque_bera,
        "jarque_bera_pvalue": jarque_bera_pvalue(jarque_bera),
    }


def summarize_complex_gaussianity(values: torch.Tensor) -> dict[str, float | int]:
    values = values.reshape(-1).to(torch.complex64)
    mask = torch.isfinite(values.real) & torch.isfinite(values.imag)
    values = values[mask]
    n = int(values.numel())
    if n == 0:
        return {
            "n": 0,
            "mean_real": float("nan"),
            "mean_imag": float("nan"),
            "std_real": float("nan"),
            "std_imag": float("nan"),
            "skew_real": float("nan"),
            "skew_imag": float("nan"),
            "excess_kurtosis_real": float("nan"),
            "excess_kurtosis_imag": float("nan"),
            "jarque_bera_real": float("nan"),
            "jarque_bera_imag": float("nan"),
            "jarque_bera_pvalue_real": float("nan"),
            "jarque_bera_pvalue_imag": float("nan"),
            "cov_real_imag": float("nan"),
            "corr_real_imag": float("nan"),
            "mahalanobis_d2_mean": float("nan"),
            "mahalanobis_d2_var": float("nan"),
            "mardia_kurtosis": float("nan"),
            "mardia_kurtosis_excess": float("nan"),
            "mahalanobis_chi2_ks": float("nan"),
        }

    real_stats = summarize_univariate_normality(values.real)
    imag_stats = summarize_univariate_normality(values.imag)
    xy = torch.stack((values.real.to(torch.float64), values.imag.to(torch.float64)), dim=1)
    mean = xy.mean(dim=0)

    cov_real_imag = float("nan")
    corr_real_imag = float("nan")
    d2_mean = float("nan")
    d2_var = float("nan")
    mardia_kurtosis = float("nan")
    mardia_kurtosis_excess = float("nan")
    chi2_ks = float("nan")

    if n >= 2:
        centered = xy - mean
        cov = centered.T.matmul(centered) / max(n - 1, 1)
        var_real = float(cov[0, 0].item())
        var_imag = float(cov[1, 1].item())
        cov_real_imag = float(cov[0, 1].item())
        if var_real > EPS and var_imag > EPS:
            corr_real_imag = cov_real_imag / math.sqrt(var_real * var_imag)
            inv_cov = torch.linalg.pinv(cov)
            d2 = (centered.matmul(inv_cov) * centered).sum(dim=1)
            d2 = d2[torch.isfinite(d2)]
            if d2.numel() > 0:
                d2_mean = float(d2.mean().item())
                d2_var = float(d2.var(unbiased=False).item()) if d2.numel() > 1 else 0.0
                mardia_kurtosis = float((d2**2).mean().item())
                mardia_kurtosis_excess = mardia_kurtosis - 8.0
                sorted_d2 = torch.sort(d2).values
                empirical_cdf = torch.arange(
                    1,
                    sorted_d2.numel() + 1,
                    dtype=torch.float64,
                    device=sorted_d2.device,
                ) / sorted_d2.numel()
                chi2_cdf = 1.0 - torch.exp(-0.5 * sorted_d2)
                chi2_ks = float(torch.max(torch.abs(empirical_cdf - chi2_cdf)).item())

    return {
        "n": n,
        "mean_real": float(mean[0].item()),
        "mean_imag": float(mean[1].item()),
        "std_real": float(real_stats["std"]),
        "std_imag": float(imag_stats["std"]),
        "skew_real": float(real_stats["skewness"]),
        "skew_imag": float(imag_stats["skewness"]),
        "excess_kurtosis_real": float(real_stats["excess_kurtosis"]),
        "excess_kurtosis_imag": float(imag_stats["excess_kurtosis"]),
        "jarque_bera_real": float(real_stats["jarque_bera"]),
        "jarque_bera_imag": float(imag_stats["jarque_bera"]),
        "jarque_bera_pvalue_real": float(real_stats["jarque_bera_pvalue"]),
        "jarque_bera_pvalue_imag": float(imag_stats["jarque_bera_pvalue"]),
        "cov_real_imag": cov_real_imag,
        "corr_real_imag": corr_real_imag,
        "mahalanobis_d2_mean": d2_mean,
        "mahalanobis_d2_var": d2_var,
        "mardia_kurtosis": mardia_kurtosis,
        "mardia_kurtosis_excess": mardia_kurtosis_excess,
        "mahalanobis_chi2_ks": chi2_ks,
    }


def plot_hist_with_gaussian(ax, values: torch.Tensor, title: str, color: str, max_points: int, bins: int) -> None:
    stats = summarize_univariate_normality(values)
    samples = finite_1d(values).cpu()
    if max_points > 0 and samples.numel() > max_points:
        indices = torch.linspace(0, samples.numel() - 1, steps=max_points).round().to(torch.long)
        samples = torch.sort(samples).values[indices]

    if samples.numel() == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no finite samples", transform=ax.transAxes, ha="center", va="center")
        return

    ax.hist(samples.cpu().numpy(), bins=max(bins, 1), density=True, color=color, alpha=0.42, edgecolor="none")
    mean = float(stats["mean"])
    std = float(stats["std"])
    if math.isfinite(std) and std > EPS:
        lower = min(float(samples.min().item()), mean - 4.0 * std)
        upper = max(float(samples.max().item()), mean + 4.0 * std)
        xs = torch.linspace(lower, upper, steps=320, dtype=torch.float64)
        ys = torch.exp(-0.5 * ((xs - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))
        ax.plot(xs.cpu().numpy(), ys.cpu().numpy(), color="black", linewidth=1.3)
    ax.set_title(
        f"{title}\nmu={mean:.3g} sigma={std:.3g} JB p={format_pvalue(float(stats['jarque_bera_pvalue']))}",
        fontsize=9,
    )
    ax.grid(alpha=0.14)
    ax.tick_params(labelsize=8)


def normal_qq_points(values: torch.Tensor, max_points: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    samples = torch.sort(finite_1d(values).cpu()).values
    n = int(samples.numel())
    if n < 2:
        return None

    mean = float(samples.mean().item())
    std = float(samples.std(unbiased=False).item())
    if not math.isfinite(std) or std <= EPS:
        return None

    if max_points > 0 and n > max_points:
        indices = torch.linspace(0, n - 1, steps=max_points).round().to(torch.long)
        indices = torch.unique(indices, sorted=True)
    else:
        indices = torch.arange(n)

    probs = ((indices.to(torch.float64) + 0.5) / n).tolist()
    theoretical = torch.tensor([NORMAL.inv_cdf(prob) * std + mean for prob in probs], dtype=torch.float64)
    observed = samples[indices]
    return theoretical, observed


def plot_normal_qq(ax, values: torch.Tensor, title: str, color: str, max_points: int) -> None:
    points = normal_qq_points(values, max_points)
    if points is None:
        ax.set_title(title)
        ax.text(0.5, 0.5, "not enough variance", transform=ax.transAxes, ha="center", va="center")
        return

    theoretical, observed = points
    ax.scatter(theoretical.cpu().numpy(), observed.cpu().numpy(), s=7, alpha=0.4, linewidths=0, color=color)
    lower = min(float(theoretical.min().item()), float(observed.min().item()))
    upper = max(float(theoretical.max().item()), float(observed.max().item()))
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.0, alpha=0.75)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.14)
    ax.tick_params(labelsize=8)

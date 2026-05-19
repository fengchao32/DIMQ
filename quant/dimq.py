"""Core math utilities for DIMQ weight quantization.

The implementation follows the local algorithm spec:
learn one non-uniform codebook per layer, optimize a Gibbs softmin
distortion term during QAT, and use nearest-center hard quantization for
fake-quant forward/export.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass
class DIMQConfig:
    w_bits: int = 3
    a_bits: int | None = None
    a_center_init: str = "kmeans"
    a_cluster_momentum: float = 0.95
    a_kmeans_iters: int = 10
    a_kmeans_sample_size: int = 32768
    target_modules: tuple[str, ...] = ("Conv2d", "Linear")
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    skip_first: bool = True
    skip_last: bool = True
    skip_downsample: bool = True
    skip_depthwise: bool = False
    skip_norm: bool = True
    quantize_bias: bool = False

    tau_start: float = 1.0
    tau_end: float = 1e-5
    tau_schedule: str = "exponential"
    total_epochs: int = 80

    lambda_dimq: float = 1e-4
    gamma_sep: float = 1e-3
    eta_margin: float = 1.0
    sep_eps: float = 1e-12

    center_init: str = "kmeans"
    center_lr_scale: float = 1.0
    loss_reduction: str = "sum"
    chunk_size: int = 262144

    forward_mode: str = "hard_ste"
    sort_centers_after_step: bool = False
    kmeans_iters: int = 50
    kmeans_sample_size: int = 200000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(cfg: DIMQConfig | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_tau(progress: float, cfg: DIMQConfig | Mapping[str, Any]) -> float:
    """Return the annealed DIMQ temperature for progress in [0, 1]."""

    progress = min(1.0, max(0.0, float(progress)))
    tau_start = float(_cfg_get(cfg, "tau_start", 1.0))
    tau_end = float(_cfg_get(cfg, "tau_end", 1e-5))
    schedule = _cfg_get(cfg, "tau_schedule", "exponential")

    tau_start = max(tau_start, 1e-12)
    tau_end = max(tau_end, 1e-12)

    if schedule == "exponential":
        return tau_start * (tau_end / tau_start) ** progress
    if schedule == "linear":
        return tau_start - progress * (tau_start - tau_end)
    if schedule == "cosine":
        return tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported tau schedule: {schedule}")


@torch.no_grad()
def init_centers_1d(
    w: Tensor,
    K: int,
    method: str = "kmeans",
    *,
    kmeans_iters: int = 50,
    sample_size: int = 200000,
) -> Tensor:
    """Initialize a sorted 1D codebook from a layer weight tensor."""

    if K <= 0:
        raise ValueError("K must be positive")

    flat = w.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("Cannot initialize centers from an empty tensor")

    finite = torch.isfinite(flat)
    if not bool(finite.all()):
        flat = flat[finite]
    if flat.numel() == 0:
        raise ValueError("Cannot initialize centers because all weights are non-finite")

    if flat.numel() > sample_size:
        perm = torch.randperm(flat.numel(), device=flat.device)[:sample_size]
        sample = flat[perm]
    else:
        sample = flat

    method = method.lower()
    if method == "quantile":
        centers = _quantile_centers(sample, K)
    elif method == "linspace_std":
        centers = _linspace_std_centers(sample, K)
    elif method == "linspace_minmax":
        centers = _linspace_minmax_centers(sample, K)
    elif method == "kmeans":
        centers = _kmeans_1d(sample, K, kmeans_iters)
    else:
        raise ValueError(f"Unsupported center initialization: {method}")

    centers = _postprocess_centers(centers, sample, K)
    return centers.to(device=w.device, dtype=w.dtype)


def _quantile_centers(flat: Tensor, K: int) -> Tensor:
    if flat.numel() == 1:
        return flat.new_full((K,), flat.item())
    q = torch.linspace(0.0, 1.0, K + 2, device=flat.device, dtype=flat.dtype)[1:-1]
    return torch.quantile(flat, q)


def _linspace_std_centers(flat: Tensor, K: int) -> Tensor:
    mu = flat.mean()
    sigma = flat.std(unbiased=False)
    if float(sigma) <= 0.0:
        return flat.new_full((K,), float(mu))
    return torch.linspace(mu - 3.0 * sigma, mu + 3.0 * sigma, K, device=flat.device, dtype=flat.dtype)


def _linspace_minmax_centers(flat: Tensor, K: int) -> Tensor:
    lo = flat.min()
    hi = flat.max()
    if float((hi - lo).abs()) <= 0.0:
        return flat.new_full((K,), float(lo))
    return torch.linspace(lo, hi, K, device=flat.device, dtype=flat.dtype)


def _kmeans_1d(flat: Tensor, K: int, iters: int) -> Tensor:
    centers = _quantile_centers(flat, K)
    if K == 1:
        return centers

    for _ in range(max(1, int(iters))):
        dist = (flat[:, None] - centers[None, :]).pow(2)
        assignment = dist.argmin(dim=1)
        new_centers = centers.clone()
        for k in range(K):
            mask = assignment == k
            if bool(mask.any()):
                new_centers[k] = flat[mask].mean()
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers.sort().values
    return centers


def _postprocess_centers(centers: Tensor, reference: Tensor, K: int) -> Tensor:
    if centers.numel() != K:
        raise ValueError(f"Expected {K} centers, got {centers.numel()}")
    if not bool(torch.isfinite(centers).all()):
        centers = _linspace_std_centers(reference, K)
    centers = centers.float().sort().values

    span = (reference.max() - reference.min()).abs()
    sigma = reference.std(unbiased=False).abs()
    scale = torch.maximum(span, sigma).clamp_min(1e-6)
    min_gap = scale * 1e-6

    fixed = centers.clone()
    for i in range(1, K):
        if fixed[i] <= fixed[i - 1]:
            fixed[i] = fixed[i - 1] + min_gap

    center_shift = fixed.mean() - centers.mean()
    fixed = fixed - center_shift
    return fixed.sort().values


def hard_quantize_weight(
    w: Tensor,
    centers: Tensor,
    chunk_size: int | None = 262144,
) -> tuple[Tensor, Tensor]:
    """Nearest-center hard quantization with optional chunking."""

    flat = w.float().reshape(-1)
    c = centers.float().reshape(-1)
    if c.numel() == 0:
        raise ValueError("centers must be non-empty")

    chunks = _chunk_ranges(flat.numel(), chunk_size)
    idx_list: list[Tensor] = []
    val_list: list[Tensor] = []
    for start, end in chunks:
        part = flat[start:end]
        dist = (part[:, None] - c[None, :]).pow(2)
        idx = dist.argmin(dim=1)
        idx_list.append(idx)
        val_list.append(centers.reshape(-1)[idx].to(dtype=w.dtype))

    idx = torch.cat(idx_list, dim=0)
    values = torch.cat(val_list, dim=0).reshape_as(w)
    return values, idx.reshape(w.shape)


def soft_dequantize_weight(
    w: Tensor,
    centers: Tensor,
    tau: float,
    chunk_size: int | None = 262144,
) -> Tensor:
    """Differentiable soft dequantization used for optional warmup."""

    tau = max(float(tau), 1e-12)
    flat = w.float().reshape(-1)
    c = centers.float().reshape(-1)
    val_list: list[Tensor] = []
    for start, end in _chunk_ranges(flat.numel(), chunk_size):
        part = flat[start:end]
        dist = (part[:, None] - c[None, :]).pow(2)
        probs = torch.softmax(-dist / tau, dim=1)
        val_list.append((probs @ c).to(dtype=w.dtype))
    return torch.cat(val_list, dim=0).reshape_as(w)


def dimq_softmin_loss(
    w: Tensor,
    centers: Tensor,
    tau: float,
    reduction: str = "sum",
    chunk_size: int | None = 262144,
) -> Tensor:
    """Gibbs softmin distortion term from the DIMQ objective."""

    tau = max(float(tau), 1e-12)
    flat = w.float().reshape(-1)
    c = centers.float().reshape(-1)
    K = c.numel()
    if K == 0:
        raise ValueError("centers must be non-empty")

    total = flat.new_tensor(0.0)
    count = 0
    for start, end in _chunk_ranges(flat.numel(), chunk_size):
        part = flat[start:end]
        dist = (part[:, None] - c[None, :]).pow(2)
        softmin = -tau * (torch.logsumexp(-dist / tau, dim=1) - math.log(K))
        total = total + softmin.sum()
        count += part.numel()

    if reduction == "sum":
        return total
    if reduction == "mean_per_layer":
        return total / max(1, count)
    raise ValueError(f"Unsupported loss reduction: {reduction}")


def separation_loss(
    w: Tensor,
    centers: Tensor,
    eta: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Hinge margin regularizer that discourages codebook collapse."""

    c = centers.float().reshape(-1)
    K = c.numel()
    if K <= 1:
        return c.new_tensor(0.0)

    sigma = w.detach().float().std(unbiased=False).clamp_min(1e-12)
    delta = float(eta) * 6.0 * sigma / K

    diff = c[:, None] - c[None, :]
    dist = torch.sqrt(diff.pow(2) + float(eps))
    mask = torch.triu(torch.ones(K, K, device=c.device, dtype=torch.bool), diagonal=1)
    pair_dist = dist[mask]
    return torch.relu(delta - pair_dist).pow(2).sum()


@torch.no_grad()
def soft_assignment_stats(
    w: Tensor,
    centers: Tensor,
    tau: float,
    chunk_size: int | None = 262144,
) -> dict[str, Tensor]:
    """Return diagnostic statistics for a DIMQ layer."""

    tau = max(float(tau), 1e-12)
    flat = w.detach().float().reshape(-1)
    c = centers.detach().float().reshape(-1)
    K = c.numel()
    entropy_sum = flat.new_tensor(0.0)
    hard_counts = torch.zeros(K, device=flat.device, dtype=torch.long)
    total = 0

    for start, end in _chunk_ranges(flat.numel(), chunk_size):
        part = flat[start:end]
        dist = (part[:, None] - c[None, :]).pow(2)
        probs = torch.softmax(-dist / tau, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        entropy_sum = entropy_sum + entropy.sum()
        idx = dist.argmin(dim=1)
        hard_counts += torch.bincount(idx, minlength=K)
        total += part.numel()

    sigma = flat.std(unbiased=False).clamp_min(1e-12)
    delta = 6.0 * sigma / K
    pair_dist = _pairwise_center_distances(c)
    if pair_dist.numel() == 0:
        min_dist = flat.new_tensor(0.0)
        violation_ratio = flat.new_tensor(0.0)
    else:
        min_dist = pair_dist.min()
        violation_ratio = (pair_dist < delta).float().mean()

    usage = hard_counts.float() / max(1, total)
    return {
        "avg_assignment_entropy": entropy_sum / max(1, total),
        "hard_codebook_usage": usage,
        "center_min": c.min(),
        "center_max": c.max(),
        "center_pair_min_distance": min_dist,
        "margin_delta": delta,
        "sep_violation_ratio": violation_ratio,
        "unique_quant_values_after_hard": (hard_counts > 0).sum(),
    }


def _pairwise_center_distances(c: Tensor) -> Tensor:
    K = c.numel()
    if K <= 1:
        return c.new_empty((0,))
    diff = (c[:, None] - c[None, :]).abs()
    mask = torch.triu(torch.ones(K, K, device=c.device, dtype=torch.bool), diagonal=1)
    return diff[mask]


def _chunk_ranges(numel: int, chunk_size: int | None) -> list[tuple[int, int]]:
    if numel < 0:
        raise ValueError("numel must be non-negative")
    if numel == 0:
        return [(0, 0)]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= numel:
        return [(0, numel)]
    return [(start, min(start + int(chunk_size), numel)) for start in range(0, numel, int(chunk_size))]

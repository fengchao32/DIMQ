"""DIMQ wrappers for torch Conv2d and Linear modules."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dimq import (
    DIMQConfig,
    dimq_softmin_loss,
    hard_quantize_weight,
    init_centers_1d,
    separation_loss,
    soft_assignment_stats,
    soft_dequantize_weight,
)


class _DIMQMixin:
    name: str
    cfg: DIMQConfig
    weight: nn.Parameter
    centers: nn.Parameter
    activation_centers: torch.Tensor
    activation_centers_initialized: torch.Tensor
    tau: float

    def _init_dimq_state(self, weight: nn.Parameter, cfg: DIMQConfig, name: str) -> None:
        self.name = name
        self.cfg = cfg
        self.weight = weight
        self.K = 2 ** int(cfg.w_bits)
        self.centers = nn.Parameter(torch.empty(self.K, device=weight.device, dtype=weight.dtype))
        self.tau = float(cfg.tau_start)
        self.init_centers(cfg.center_init)
        self._init_activation_quant_state(weight)

    def _init_activation_quant_state(self, weight: nn.Parameter) -> None:
        if self.cfg.a_bits is None:
            self.aK = 0
            self.register_buffer("activation_centers", torch.empty(0, device=weight.device, dtype=weight.dtype))
        else:
            self.aK = 2 ** int(self.cfg.a_bits)
            self.register_buffer(
                "activation_centers",
                torch.empty(self.aK, device=weight.device, dtype=weight.dtype),
            )
        self.register_buffer(
            "activation_centers_initialized",
            torch.tensor(False, device=weight.device, dtype=torch.bool),
        )

    @torch.no_grad()
    def init_centers(self, method: str | None = None) -> None:
        method = method or self.cfg.center_init
        centers = init_centers_1d(
            self.weight,
            self.K,
            method,
            kmeans_iters=self.cfg.kmeans_iters,
            sample_size=self.cfg.kmeans_sample_size,
        )
        self.centers.copy_(centers)

    def set_tau(self, tau: float) -> None:
        self.tau = max(float(tau), float(self.cfg.tau_end))

    def fake_quant_weight(self) -> torch.Tensor:
        if self.cfg.forward_mode == "hard_ste":
            w_hard, _ = hard_quantize_weight(self.weight, self.centers, self.cfg.chunk_size)
            return self.weight + (w_hard - self.weight).detach()
        if self.cfg.forward_mode == "soft":
            return soft_dequantize_weight(self.weight, self.centers, self.tau, self.cfg.chunk_size)
        raise ValueError(f"Unsupported DIMQ forward mode: {self.cfg.forward_mode}")

    def fake_quant_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.a_bits is None:
            return x
        self._maybe_update_activation_centers(x)
        q, _ = hard_quantize_weight(x, self.activation_centers, self.cfg.chunk_size)
        return x + (q - x).detach()

    @torch.no_grad()
    def _maybe_update_activation_centers(self, x: torch.Tensor) -> None:
        if self.cfg.a_bits is None:
            return
        sample = self._sample_activation(x.detach())
        if sample.numel() == 0:
            return
        if not bool(self.activation_centers_initialized.item()):
            centers = init_centers_1d(
                sample,
                self.aK,
                self.cfg.a_center_init,
                kmeans_iters=self.cfg.a_kmeans_iters,
                sample_size=self.cfg.a_kmeans_sample_size,
            )
            self.activation_centers.copy_(centers)
            self.activation_centers_initialized.fill_(True)
            return
        if not self.training:
            return
        momentum = float(self.cfg.a_cluster_momentum)
        if momentum >= 1.0:
            return
        batch_centers = self._batch_activation_centers(sample)
        updated = momentum * self.activation_centers.float() + (1.0 - momentum) * batch_centers.float()
        self.activation_centers.copy_(updated.sort().values.to(dtype=self.activation_centers.dtype))

    @torch.no_grad()
    def _sample_activation(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.float().reshape(-1)
        finite = torch.isfinite(flat)
        if not bool(finite.all()):
            flat = flat[finite]
        max_samples = int(self.cfg.a_kmeans_sample_size)
        if max_samples > 0 and flat.numel() > max_samples:
            index = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
            flat = flat[index]
        return flat

    @torch.no_grad()
    def _batch_activation_centers(self, sample: torch.Tensor) -> torch.Tensor:
        centers = self.activation_centers.float()
        dist = (sample.float()[:, None] - centers[None, :]).pow(2)
        assignment = dist.argmin(dim=1)
        new_centers = centers.clone()
        for index in range(self.aK):
            mask = assignment == index
            if bool(mask.any()):
                new_centers[index] = sample[mask].float().mean()
        return new_centers.sort().values

    def dimq_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dimq_distortion_loss(), self.dimq_separation_loss()

    def dimq_distortion_loss(self) -> torch.Tensor:
        distortion = dimq_softmin_loss(
            self.weight,
            self.centers,
            tau=self.tau,
            reduction=self.cfg.loss_reduction,
            chunk_size=self.cfg.chunk_size,
        )
        return distortion

    def dimq_separation_loss(self) -> torch.Tensor:
        return separation_loss(
            self.weight,
            self.centers,
            eta=self.cfg.eta_margin,
            eps=self.cfg.sep_eps,
        )

    @torch.no_grad()
    def sort_centers_(self) -> None:
        self.centers.data.copy_(self.centers.data.sort().values)

    @torch.no_grad()
    def hard_quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        return hard_quantize_weight(self.weight, self.centers, self.cfg.chunk_size)

    @torch.no_grad()
    def quantization_stats(self) -> dict[str, torch.Tensor]:
        return soft_assignment_stats(self.weight, self.centers, self.tau, self.cfg.chunk_size)


class DIMQConv2d(nn.Module, _DIMQMixin):
    """Conv2d with DIMQ fake-quantized weights."""

    def __init__(self, conv: nn.Conv2d, cfg: DIMQConfig, name: str):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self._reversed_padding_repeated_twice = conv._reversed_padding_repeated_twice
        self.bias = conv.bias
        self._init_dimq_state(conv.weight, cfg, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fake_quant_activation(x)
        weight = self.fake_quant_weight()
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, self.bias, self.stride, padding, self.dilation, self.groups)

    def to_float_module(self, quantize_weight: bool = True) -> nn.Conv2d:
        conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        weight = self.hard_quantized_weight()[0] if quantize_weight else self.weight.detach()
        conv.weight = nn.Parameter(weight.detach().clone())
        if self.bias is not None:
            conv.bias = nn.Parameter(self.bias.detach().clone())
        return conv

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, bits={self.cfg.w_bits}, "
            f"K={self.K}, name={self.name}"
        )


class DIMQLinear(nn.Module, _DIMQMixin):
    """Linear layer with DIMQ fake-quantized weights."""

    def __init__(self, linear: nn.Linear, cfg: DIMQConfig, name: str):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias
        self._init_dimq_state(linear.weight, cfg, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fake_quant_activation(x)
        weight = self.fake_quant_weight()
        return F.linear(x, weight, self.bias)

    def to_float_module(self, quantize_weight: bool = True) -> nn.Linear:
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        weight = self.hard_quantized_weight()[0] if quantize_weight else self.weight.detach()
        linear.weight = nn.Parameter(weight.detach().clone())
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.detach().clone())
        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.cfg.w_bits}, K={self.K}, name={self.name}"
        )


DIMQModule = DIMQConv2d | DIMQLinear


def collect_quant_layers(model: nn.Module, cfg: DIMQConfig) -> list[tuple[str, nn.Module]]:
    """Return named Conv2d/Linear layers selected for DIMQ wrapping."""

    target_names = set(cfg.target_modules)
    selected: list[tuple[str, nn.Module]] = []
    K = 2 ** int(cfg.w_bits)

    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, (DIMQConv2d, DIMQLinear)):
            continue
        if module.__class__.__name__ not in target_names:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if module.weight.numel() < K:
            continue
        if cfg.skip_downsample and _is_downsample_layer(name):
            continue
        if cfg.skip_depthwise and _is_depthwise_conv(module):
            continue
        if cfg.include_patterns and not _matches_any(name, cfg.include_patterns):
            continue
        if cfg.exclude_patterns and _matches_any(name, cfg.exclude_patterns):
            continue
        selected.append((name, module))

    if cfg.skip_first and selected:
        selected = selected[1:]
    if cfg.skip_last and selected:
        selected = selected[:-1]
    return selected


def replace_with_dimq_wrapper(
    model: nn.Module,
    layer_name: str,
    module: nn.Module,
    cfg: DIMQConfig,
) -> DIMQModule:
    """Replace one module in-place and return its DIMQ wrapper."""

    if isinstance(module, nn.Conv2d):
        wrapper: DIMQModule = DIMQConv2d(module, cfg, layer_name)
    elif isinstance(module, nn.Linear):
        wrapper = DIMQLinear(module, cfg, layer_name)
    else:
        raise TypeError(f"DIMQ can only wrap Conv2d/Linear, got {type(module)!r}")

    parent, leaf = _resolve_parent(model, layer_name)
    parent._modules[leaf] = wrapper
    return wrapper


def apply_dimq(model: nn.Module, cfg: DIMQConfig) -> list[DIMQModule]:
    """Collect and wrap all configured DIMQ layers in-place."""

    layers = collect_quant_layers(model, cfg)
    wrappers: list[DIMQModule] = []
    for name, module in layers:
        wrappers.append(replace_with_dimq_wrapper(model, name, module, cfg))
    return wrappers


def get_dimq_modules(model: nn.Module) -> list[tuple[str, DIMQModule]]:
    modules: list[tuple[str, DIMQModule]] = []
    for name, module in model.named_modules():
        if isinstance(module, (DIMQConv2d, DIMQLinear)):
            modules.append((name, module))
    return modules


def dimq_regularization_loss(
    modules: Iterable[DIMQModule],
    *,
    compute_dimq: bool = True,
    compute_sep: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum DIMQ distortion and separation losses over wrapped modules."""

    modules = list(modules)
    if not modules:
        raise ValueError("No DIMQ modules were provided")
    device = modules[0].weight.device
    distortion = torch.zeros((), device=device)
    separation = torch.zeros((), device=device)
    for module in modules:
        if compute_dimq:
            distortion = distortion + module.dimq_distortion_loss()
        if compute_sep:
            separation = separation + module.dimq_separation_loss()
    return distortion, separation


def set_dimq_tau(modules: Iterable[DIMQModule], tau: float) -> None:
    for module in modules:
        module.set_tau(tau)


@torch.no_grad()
def sort_all_centers_(modules: Iterable[DIMQModule]) -> None:
    for module in modules:
        module.sort_centers_()


def _resolve_parent(model: nn.Module, layer_name: str) -> tuple[nn.Module, str]:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _is_downsample_layer(layer_name: str) -> bool:
    return "downsample" in layer_name.split(".")


def _is_depthwise_conv(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.groups == module.in_channels
        and module.in_channels == module.out_channels
    )


def _matches_any(layer_name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(layer_name, pattern) for pattern in patterns)

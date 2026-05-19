"""PyTorch reference implementation for DIMQ packed-index inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .export import unpack_u4_indices


def quantize_activation_u4(
    x: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
) -> torch.Tensor:
    """Uniform affine 4-bit activation quantization."""

    if float(act_scale) <= 0.0:
        raise ValueError("act_scale must be positive")
    q = torch.round(x.float() / float(act_scale)) + int(act_zero_point)
    return torch.clamp(q, 0, 15).to(torch.uint8)


def dequantize_activation_u4(
    q: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
) -> torch.Tensor:
    return float(act_scale) * (q.float() - int(act_zero_point))


def build_product_table(
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
) -> torch.Tensor:
    """Return ``P[w_idx, a_idx] = C[w_idx] * s_A * (a_idx - z_A)``."""

    centers = codebook.float().reshape(-1)
    if centers.numel() != 16:
        raise ValueError(f"expected 16 codebook entries, got {centers.numel()}")
    a = torch.arange(16, device=centers.device, dtype=torch.float32)
    a_hat = float(act_scale) * (a - int(act_zero_point))
    return centers[:, None] * a_hat[None, :]


def dequantize_packed_weight(
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    weight_shape: tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Materialize a dequantized weight tensor for reference checks only."""

    numel = int(torch.tensor(tuple(weight_shape)).prod().item())
    indices = unpack_u4_indices(packed_weight, numel, shape=weight_shape).long()
    return codebook.float().reshape(-1)[indices]


def dimq_linear_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference Linear using packed weight indices and uniform affine A4."""

    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dim {in_features}, got {x.shape[-1]}")
    original_shape = x.shape[:-1]
    x_2d = x.reshape(-1, in_features)
    w_idx = unpack_u4_indices(
        packed_weight,
        int(out_features) * int(in_features),
        shape=(int(out_features), int(in_features)),
    ).long()
    table = build_product_table(codebook.to(device=x.device), act_scale, act_zero_point)
    a_idx = quantize_activation_u4(x_2d, act_scale, act_zero_point).long()

    y = x_2d.new_zeros((x_2d.shape[0], int(out_features)), dtype=torch.float32)
    for n in range(int(out_features)):
        y[:, n] = table[w_idx[n].to(device=x.device), a_idx].sum(dim=1)
    if bias is not None:
        y = y + bias.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    return y.reshape(*original_shape, int(out_features))


def dimq_linear_dequant_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equivalent reference that explicitly dequantizes activations and weights."""

    w_hat = dequantize_packed_weight(
        packed_weight,
        codebook.to(device=x.device),
        (int(out_features), int(in_features)),
    )
    a_idx = quantize_activation_u4(x, act_scale, act_zero_point)
    x_hat = dequantize_activation_u4(a_idx, act_scale, act_zero_point).to(device=x.device)
    return F.linear(x_hat.float(), w_hat.float(), None if bias is None else bias.float().to(x.device))


def dimq_conv2d_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    weight_shape: tuple[int, int, int, int] | list[int],
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Reference Conv2d using codebook lookup and uniform affine A4."""

    weight_shape = tuple(int(v) for v in weight_shape)
    w_hat = dequantize_packed_weight(
        packed_weight,
        codebook.to(device=x.device),
        weight_shape,
    )
    a_idx = quantize_activation_u4(x, act_scale, act_zero_point)
    x_hat = dequantize_activation_u4(a_idx, act_scale, act_zero_point).to(device=x.device)
    return F.conv2d(
        x_hat.float(),
        w_hat.float(),
        None if bias is None else bias.float().to(x.device),
        stride,
        padding,
        dilation,
        groups,
    )


def dimq_conv2d_lut_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    weight_shape: tuple[int, int, int, int] | list[int],
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Slow product-table Conv2d reference mirroring the CUDA direct kernel."""

    if x.dim() != 4:
        raise ValueError("conv2d reference expects NCHW input")
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    out_channels, cin_per_group, kernel_h, kernel_w = [int(v) for v in weight_shape]
    batch, in_channels, in_h, in_w = [int(v) for v in x.shape]
    out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
    if out_channels % int(groups) != 0:
        raise ValueError("out_channels must be divisible by groups")
    if in_channels != cin_per_group * int(groups):
        raise ValueError("input channel count does not match weight_shape/groups")

    w_idx = unpack_u4_indices(
        packed_weight,
        out_channels * cin_per_group * kernel_h * kernel_w,
        shape=weight_shape,
    ).long().to(device=x.device)
    a_idx = quantize_activation_u4(x, act_scale, act_zero_point).long()
    table = build_product_table(codebook.to(device=x.device), act_scale, act_zero_point)
    out = x.new_zeros((batch, out_channels, out_h, out_w), dtype=torch.float32)
    cout_per_group = out_channels // int(groups)
    for n in range(batch):
        for co in range(out_channels):
            group = co // cout_per_group
            for oh in range(out_h):
                for ow in range(out_w):
                    acc = x.new_tensor(0.0, dtype=torch.float32)
                    for ci_g in range(cin_per_group):
                        ci = group * cin_per_group + ci_g
                        for kh in range(kernel_h):
                            ih = oh * stride_h + kh * dil_h - pad_h
                            if ih < 0 or ih >= in_h:
                                continue
                            for kw in range(kernel_w):
                                iw = ow * stride_w + kw * dil_w - pad_w
                                if iw < 0 or iw >= in_w:
                                    continue
                                acc = acc + table[w_idx[co, ci_g, kh, kw], a_idx[n, ci, ih, iw]]
                    if bias is not None:
                        acc = acc + bias.to(device=x.device, dtype=torch.float32)[co]
                    out[n, co, oh, ow] = acc
    return out


def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"expected pair, got {value!r}")
    return int(value[0]), int(value[1])

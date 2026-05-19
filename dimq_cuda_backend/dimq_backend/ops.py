"""Python entry points for DIMQ custom CUDA operators."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Sequence

import torch


@lru_cache(maxsize=1)
def _extension_available() -> bool:
    try:
        importlib.import_module("dimq_backend._C")
    except Exception:
        return False
    return True


def is_available() -> bool:
    """Return whether the compiled CUDA extension is importable."""

    return _extension_available()


def require_extension() -> None:
    if not is_available():
        raise RuntimeError(
            "dimq_backend CUDA extension is not available. "
            "Build it with `cd dimq_cuda_backend && python setup.py build_ext --inplace` "
            "or install the package with `pip install -e dimq_cuda_backend`."
        )


def pack_u4(indices: torch.Tensor) -> torch.Tensor:
    require_extension()
    return torch.ops.dimq.pack_u4(indices)


def unpack_u4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    require_extension()
    return torch.ops.dimq.unpack_u4(packed, int(numel))


def quantize_activation_u4(x: torch.Tensor, act_scale: float, act_zero_point: int) -> torch.Tensor:
    require_extension()
    return torch.ops.dimq.quantize_activation_u4(x, float(act_scale), int(act_zero_point))


def build_product_table(
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
) -> torch.Tensor:
    require_extension()
    return torch.ops.dimq.build_product_table(codebook, float(act_scale), int(act_zero_point))


def dimq_linear_lut(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    bias: torch.Tensor | None,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    require_extension()
    return torch.ops.dimq.linear_lut(
        x,
        packed_weight,
        codebook,
        float(act_scale),
        int(act_zero_point),
        bias,
        int(out_features),
        int(in_features),
    )


def dimq_conv2d_lut(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    codebook: torch.Tensor,
    act_scale: float,
    act_zero_point: int,
    bias: torch.Tensor | None,
    weight_shape: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    require_extension()
    out_channels, cin_per_group, kernel_h, kernel_w = [int(v) for v in weight_shape]
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    return torch.ops.dimq.conv2d_lut(
        x,
        packed_weight,
        codebook,
        float(act_scale),
        int(act_zero_point),
        bias,
        out_channels,
        cin_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        int(groups),
    )


def _pair(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"expected pair, got {value!r}")
    return int(value[0]), int(value[1])

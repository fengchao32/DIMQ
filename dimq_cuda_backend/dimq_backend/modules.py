"""Runtime wrappers for packed DIMQ layers."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from . import ops
from .reference import dimq_conv2d_reference, dimq_linear_reference


class DIMQLinear(nn.Module):
    """Linear layer backed by packed u4 weight indices and a learned codebook."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        codebook: torch.Tensor,
        act_scale: float,
        act_zero_point: int,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None = None,
        *,
        use_cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight.contiguous().to(torch.uint8))
        self.register_buffer("codebook", codebook.contiguous())
        self.register_buffer("bias", None if bias is None else bias.contiguous())
        self.act_scale = float(act_scale)
        self.act_zero_point = int(act_zero_point)
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.use_cuda_kernel = bool(use_cuda_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_kernel and x.is_cuda and self.packed_weight.is_cuda and ops.is_available():
            return ops.dimq_linear_lut(
                x,
                self.packed_weight,
                self.codebook,
                self.act_scale,
                self.act_zero_point,
                self.bias,
                self.out_features,
                self.in_features,
            )
        return dimq_linear_reference(
            x,
            self.packed_weight,
            self.codebook,
            self.act_scale,
            self.act_zero_point,
            self.out_features,
            self.in_features,
            self.bias,
        )

    @classmethod
    def from_export_layer(cls, layer: Mapping[str, Any], *, use_cuda_kernel: bool = True) -> "DIMQLinear":
        return cls(
            layer["packed_weight"],
            layer["codebook"],
            layer["act_scale"],
            layer["act_zero_point"],
            layer["out_features"],
            layer["in_features"],
            layer.get("bias"),
            use_cuda_kernel=use_cuda_kernel,
        )


class DIMQConv2d(nn.Module):
    """Conv2d layer backed by packed u4 weight indices and a learned codebook."""

    def __init__(
        self,
        packed_weight: torch.Tensor,
        codebook: torch.Tensor,
        act_scale: float,
        act_zero_point: int,
        weight_shape: tuple[int, int, int, int] | list[int],
        bias: torch.Tensor | None = None,
        *,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 0,
        dilation: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        use_cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight.contiguous().to(torch.uint8))
        self.register_buffer("codebook", codebook.contiguous())
        self.register_buffer("bias", None if bias is None else bias.contiguous())
        self.act_scale = float(act_scale)
        self.act_zero_point = int(act_zero_point)
        self.weight_shape = tuple(int(v) for v in weight_shape)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.use_cuda_kernel = bool(use_cuda_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_kernel and x.is_cuda and self.packed_weight.is_cuda and ops.is_available():
            return ops.dimq_conv2d_lut(
                x,
                self.packed_weight,
                self.codebook,
                self.act_scale,
                self.act_zero_point,
                self.bias,
                self.weight_shape,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return dimq_conv2d_reference(
            x,
            self.packed_weight,
            self.codebook,
            self.act_scale,
            self.act_zero_point,
            self.weight_shape,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @classmethod
    def from_export_layer(cls, layer: Mapping[str, Any], *, use_cuda_kernel: bool = True) -> "DIMQConv2d":
        return cls(
            layer["packed_weight"],
            layer["codebook"],
            layer["act_scale"],
            layer["act_zero_point"],
            layer["weight_shape"],
            layer.get("bias"),
            stride=layer.get("stride", (1, 1)),
            padding=layer.get("padding", (0, 0)),
            dilation=layer.get("dilation", (1, 1)),
            groups=layer.get("groups", 1),
            use_cuda_kernel=use_cuda_kernel,
        )


def make_packed_layer(layer: Mapping[str, Any], *, use_cuda_kernel: bool = True) -> nn.Module:
    if layer["type"] == "linear":
        return DIMQLinear.from_export_layer(layer, use_cuda_kernel=use_cuda_kernel)
    if layer["type"] == "conv2d":
        return DIMQConv2d.from_export_layer(layer, use_cuda_kernel=use_cuda_kernel)
    raise ValueError(f"unsupported packed DIMQ layer type {layer['type']!r}")


def replace_modules_with_packed_dimq(
    model: nn.Module,
    packed_state: Mapping[str, Any],
    *,
    use_cuda_kernel: bool = True,
) -> nn.Module:
    """Replace matching modules in ``model`` with packed DIMQ runtime wrappers."""

    for name, layer in packed_state["layers"].items():
        parent, leaf = _resolve_parent(model, name)
        parent._modules[leaf] = make_packed_layer(layer, use_cuda_kernel=use_cuda_kernel)
    return model


def _resolve_parent(model: nn.Module, layer_name: str) -> tuple[nn.Module, str]:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"expected pair, got {value!r}")
    return int(value[0]), int(value[1])

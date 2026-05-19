"""Mask R-CNN with a torchvision Swin-S backbone and FPN.

Torchvision exposes Swin-S as an image classifier and MaskRCNN as a generic
detection model, but it does not provide a ready-made Swin-S Mask R-CNN
builder. This module bridges those two pieces while keeping the model as plain
PyTorch modules so the existing DIMQ Conv2d/Linear wrappers can be reused.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import Swin_S_Weights, swin_s
from torchvision.models.detection import MaskRCNN
from torchvision.models.swin_transformer import ShiftedWindowAttention, shifted_window_attention
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


SWIN_S_STAGE_CHANNELS = [96, 192, 384, 768]


class SwinSBackboneWithFPN(nn.Module):
    """Swin-S stage outputs adapted to a torchvision FPN backbone."""

    out_channels: int

    def __init__(
        self,
        *,
        weights: Swin_S_Weights | None = None,
        fpn_out_channels: int = 256,
    ) -> None:
        super().__init__()
        swin = swin_s(weights=weights)
        self.body = swin.features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=SWIN_S_STAGE_CHANNELS,
            out_channels=fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = int(fpn_out_channels)

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        features: OrderedDict[str, Tensor] = OrderedDict()

        x = self.body[0](x)
        x = self.body[1](x)
        features["0"] = _nhwc_to_nchw(x)

        x = self.body[2](x)
        x = self.body[3](x)
        features["1"] = _nhwc_to_nchw(x)

        x = self.body[4](x)
        x = self.body[5](x)
        features["2"] = _nhwc_to_nchw(x)

        x = self.body[6](x)
        x = self.body[7](x)
        features["3"] = _nhwc_to_nchw(x)

        return self.fpn(features)


def build_maskrcnn_swin_s(
    *,
    num_classes: int,
    pretrained_backbone: bool = True,
    weights_name: str = "DEFAULT",
    fpn_out_channels: int = 256,
    min_size: int = 800,
    max_size: int = 1333,
    **kwargs: Any,
) -> MaskRCNN:
    """Build Mask R-CNN with Swin-S ImageNet initialization for the backbone.

    ``num_classes`` follows torchvision detection convention: include the
    background class at index 0, so COCO instance segmentation uses 91.
    """

    weights = resolve_swin_s_weights(weights_name) if pretrained_backbone else None
    backbone = SwinSBackboneWithFPN(weights=weights, fpn_out_channels=fpn_out_channels)
    return MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        **kwargs,
    )


def resolve_swin_s_weights(weights_name: str) -> Swin_S_Weights:
    if weights_name.upper() == "DEFAULT":
        return Swin_S_Weights.DEFAULT
    if hasattr(Swin_S_Weights, "__members__") and weights_name in Swin_S_Weights.__members__:
        return Swin_S_Weights[weights_name]
    raise ValueError(
        f"Unknown Swin-S weights {weights_name!r}; "
        f"use DEFAULT or one of {list(Swin_S_Weights.__members__)}"
    )


def enable_dimq_swin_attention_fake_quant(model: nn.Module) -> int:
    """Patch torchvision Swin attention to read fake-quantized DIMQ weights.

    Torchvision's ``ShiftedWindowAttention.forward`` calls
    ``shifted_window_attention`` with ``self.qkv.weight`` and
    ``self.proj.weight`` directly. Replacing those Linear layers with DIMQ
    wrappers is not enough, because the wrapper's ``forward`` is bypassed.
    This patch preserves torchvision's functional attention path but asks DIMQ
    wrappers for ``fake_quant_weight()`` when present.
    """

    patched = 0
    for module in model.modules():
        if not isinstance(module, ShiftedWindowAttention):
            continue
        if getattr(module, "_dimq_fake_quant_forward_enabled", False):
            continue
        module.forward = MethodType(_dimq_shifted_window_attention_forward, module)
        module._dimq_fake_quant_forward_enabled = True
        patched += 1
    return patched


def _dimq_shifted_window_attention_forward(self: ShiftedWindowAttention, x: Tensor) -> Tensor:
    relative_position_bias = self.get_relative_position_bias()
    return shifted_window_attention(
        x,
        _maybe_fake_quant_weight(self.qkv),
        _maybe_fake_quant_weight(self.proj),
        relative_position_bias,
        self.window_size,
        self.num_heads,
        shift_size=self.shift_size,
        attention_dropout=self.attention_dropout,
        dropout=self.dropout,
        qkv_bias=self.qkv.bias,
        proj_bias=self.proj.bias,
        logit_scale=getattr(self, "logit_scale", None),
        training=self.training,
    )


def _maybe_fake_quant_weight(module: nn.Module) -> Tensor:
    fake_quant_weight = getattr(module, "fake_quant_weight", None)
    if callable(fake_quant_weight):
        return fake_quant_weight()
    return module.weight


def _nhwc_to_nchw(x: Tensor) -> Tensor:
    return x.permute(0, 3, 1, 2).contiguous()

"""DIMQ packed-index CUDA/CUTLASS-style inference backend."""

from .export import (
    activation_qparams_from_minmax,
    assign_weight_indices,
    default_activation_qparams,
    export_packed_dimq_checkpoint,
    infer_module_hparams,
    pack_u4_indices,
    unpack_u4_indices,
)
from .modules import DIMQConv2d, DIMQLinear, replace_modules_with_packed_dimq
from .ops import is_available
from .reference import (
    build_product_table,
    dimq_conv2d_reference,
    dimq_linear_reference,
    quantize_activation_u4,
)

__all__ = [
    "DIMQConv2d",
    "DIMQLinear",
    "activation_qparams_from_minmax",
    "assign_weight_indices",
    "build_product_table",
    "default_activation_qparams",
    "dimq_conv2d_reference",
    "dimq_linear_reference",
    "export_packed_dimq_checkpoint",
    "infer_module_hparams",
    "is_available",
    "pack_u4_indices",
    "quantize_activation_u4",
    "replace_modules_with_packed_dimq",
    "unpack_u4_indices",
]

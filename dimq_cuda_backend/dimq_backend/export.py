"""Export utilities for packed-index DIMQ inference.

The training code in this repository saves a compact checkpoint with logical
nearest-center indices and learned per-layer codebooks.  This module converts
that representation into the runtime format used by the CUDA LUT kernels:
packed 4-bit weight indices, a 16-entry learned codebook, and fixed uniform
affine 4-bit activation quantization parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


QMIN_U4 = 0
QMAX_U4 = 15


def assign_weight_indices(weight: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Assign every weight value to the nearest learned codebook center."""

    w_flat = weight.detach().float().reshape(-1)
    centers = codebook.detach().float().reshape(-1)
    if centers.numel() > 16:
        raise ValueError(f"packed u4 export supports at most 16 centers, got {centers.numel()}")
    if centers.numel() == 0:
        raise ValueError("codebook must not be empty")

    dist = (w_flat[:, None] - centers[None, :]).pow(2)
    return dist.argmin(dim=1).to(torch.uint8).reshape(weight.shape)


def pack_u4_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack two unsigned 4-bit indices into one byte.

    The nibble order is fixed to match the CUDA helper:

    ``packed[j] = indices[2*j] | (indices[2*j + 1] << 4)``.
    """

    flat = indices.detach().contiguous().reshape(-1).to(torch.uint8)
    if flat.numel() == 0:
        return flat.new_empty((0,), dtype=torch.uint8)
    if int(flat.max().item()) > 15 or int(flat.min().item()) < 0:
        raise ValueError("u4 indices must be in [0, 15]")
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros((1,))], dim=0)
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def unpack_u4_indices(
    packed: torch.Tensor,
    numel: int,
    *,
    shape: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Unpack a byte tensor produced by :func:`pack_u4_indices`."""

    packed = packed.detach().contiguous().reshape(-1).to(torch.uint8)
    if numel < 0:
        raise ValueError("numel must be non-negative")
    needed = (numel + 1) // 2
    if packed.numel() < needed:
        raise ValueError(f"packed tensor has {packed.numel()} bytes, expected at least {needed}")
    out = torch.empty(needed * 2, device=packed.device, dtype=torch.uint8)
    out[0::2] = packed[:needed] & 0x0F
    out[1::2] = (packed[:needed] >> 4) & 0x0F
    out = out[:numel].contiguous()
    if shape is not None:
        out = out.reshape(tuple(shape))
    return out


def activation_qparams_from_minmax(
    min_val: float,
    max_val: float,
    *,
    qmin: int = QMIN_U4,
    qmax: int = QMAX_U4,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Build per-tensor uniform affine qparams from observed activation range."""

    min_val = float(min_val)
    max_val = float(max_val)
    if not torch.isfinite(torch.tensor([min_val, max_val])).all():
        raise ValueError("activation min/max must be finite")
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if max_val - min_val < eps:
        scale = 1.0
        zero_point = 0
    else:
        scale = max((max_val - min_val) / float(qmax - qmin), eps)
        zero_point = int(round(qmin - min_val / scale))
        zero_point = max(qmin, min(qmax, zero_point))
    return {
        "bit_a": 4,
        "scale": float(scale),
        "zero_point": int(zero_point),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "granularity": "per_tensor",
    }


def default_activation_qparams() -> dict[str, Any]:
    """Conservative fallback qparams for smoke tests when no calibration exists."""

    return {
        "bit_a": 4,
        "scale": 1.0,
        "zero_point": 8,
        "qmin": QMIN_U4,
        "qmax": QMAX_U4,
        "granularity": "per_tensor",
        "source": "default_uncalibrated",
    }


def load_activation_qparams(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load layer activation qparams from a JSON file."""

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("activation qparams JSON must contain a mapping")
    return {str(key): _normalize_act_qparams(value) for key, value in data.items()}


def infer_module_hparams(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Return Conv2d/Linear metadata keyed by module name."""

    hparams: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hparams[name] = {
                "type": "conv2d",
                "stride": list(_pair(module.stride)),
                "padding": list(_pair(module.padding)),
                "dilation": list(_pair(module.dilation)),
                "groups": int(module.groups),
                "bias": module.bias.detach().cpu() if module.bias is not None else None,
            }
        elif isinstance(module, nn.Linear):
            hparams[name] = {
                "type": "linear",
                "bias": module.bias.detach().cpu() if module.bias is not None else None,
            }
    return hparams


@torch.no_grad()
def export_packed_dimq_checkpoint(
    compact_checkpoint: str | Path | Mapping[str, Any],
    output_path: str | Path,
    *,
    act_qparams: Mapping[str, Mapping[str, Any]] | str | Path | None = None,
    layer_hparams: Mapping[str, Mapping[str, Any]] | None = None,
    model_name: str = "dimq_packed_lut",
    codebook_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Export a DIMQ compact checkpoint as packed u4 weights plus A4 qparams.

    ``act_qparams`` is expected to be a mapping from layer name to dictionaries
    with ``scale`` and ``zero_point``.  A ``"__default__"`` entry is accepted
    for smoke tests, but calibrated layer-wise qparams are preferred for real
    accuracy evaluation.
    """

    if isinstance(compact_checkpoint, Mapping):
        compact = dict(compact_checkpoint)
        source_path = None
    else:
        source_path = str(compact_checkpoint)
        compact = torch.load(Path(compact_checkpoint), map_location="cpu")

    if compact.get("format") != "dimq_compact":
        raise ValueError(f"expected a dimq_compact checkpoint, got {compact.get('format')!r}")

    qparams_map = _load_or_normalize_qparams(act_qparams)
    hparams_map = {str(k): dict(v) for k, v in (layer_hparams or {}).items()}
    non_quantized = compact.get("non_quantized_state_dict", {})

    layers: dict[str, dict[str, Any]] = {}
    for name, layer in compact["layers"].items():
        centers = layer["centers"].detach().float().reshape(-1)
        if centers.numel() != 16:
            raise ValueError(
                f"{name} has {centers.numel()} centers; this W4A4 backend expects 16 learned centers"
            )
        indices = layer["indices"].detach().to(torch.uint8)
        if int(indices.min().item()) < 0 or int(indices.max().item()) > 15:
            raise ValueError(f"{name} contains indices outside [0, 15]")

        shape = [int(dim) for dim in layer["shape"]]
        layer_type = _infer_layer_type(shape, hparams_map.get(name))
        bias = _find_bias(name, non_quantized, hparams_map.get(name))
        aq = _select_qparams(name, qparams_map)
        packed = pack_u4_indices(indices)

        entry: dict[str, Any] = {
            "name": name,
            "type": layer_type,
            "bit_w": 4,
            "bit_a": 4,
            "weight_shape": shape,
            "weight_numel": int(indices.numel()),
            "packed_weight": packed.cpu(),
            "packed_indices": packed.cpu(),
            "codebook": centers.to(dtype=codebook_dtype).cpu(),
            "codebook_granularity": "per_layer",
            "act_scale": float(aq["scale"]),
            "act_zero_point": int(aq["zero_point"]),
            "act_qmin": int(aq.get("qmin", QMIN_U4)),
            "act_qmax": int(aq.get("qmax", QMAX_U4)),
            "activation_qparams": aq,
            "bias": bias.detach().cpu().to(dtype=codebook_dtype) if bias is not None else None,
        }
        if layer_type == "linear":
            entry["layout"] = "linear_nk"
            entry["out_features"] = shape[0]
            entry["in_features"] = shape[1]
        else:
            hp = hparams_map.get(name, {})
            entry["layout"] = "conv_kcrs"
            entry["out_channels"] = shape[0]
            entry["in_channels_per_group"] = shape[1]
            entry["kernel_size"] = shape[2:4]
            entry["stride"] = list(_pair(hp.get("stride", (1, 1))))
            entry["padding"] = list(_pair(hp.get("padding", _default_padding(shape))))
            entry["dilation"] = list(_pair(hp.get("dilation", (1, 1))))
            entry["groups"] = int(hp.get("groups", 1))
        layers[name] = entry

    state = {
        "format": "dimq_packed_lut",
        "format_version": 1,
        "model_name": model_name,
        "source_checkpoint": source_path,
        "source_format": compact.get("format"),
        "source_extra": compact.get("extra", {}),
        "model_meta": compact.get("model_meta", {}),
        "quant_cfg": compact.get("quant_cfg", {}),
        "weight_encoding": "packed_u4_indices",
        "activation_quantization": "uniform_affine_u4",
        "layers": layers,
        "non_quantized_state_dict": non_quantized,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    return state


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, torch.Size):
        value = tuple(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"expected an int or pair, got {value!r}")


def _default_padding(shape: list[int]) -> tuple[int, int]:
    if len(shape) != 4:
        return (0, 0)
    return (int(shape[2]) // 2, int(shape[3]) // 2)


def _infer_layer_type(shape: list[int], hparams: Mapping[str, Any] | None) -> str:
    if hparams and hparams.get("type") in {"conv2d", "linear"}:
        return str(hparams["type"])
    if len(shape) == 2:
        return "linear"
    if len(shape) == 4:
        return "conv2d"
    raise ValueError(f"cannot infer layer type from weight shape {shape}")


def _find_bias(
    name: str,
    non_quantized: Mapping[str, torch.Tensor],
    hparams: Mapping[str, Any] | None,
) -> torch.Tensor | None:
    if hparams and hparams.get("bias") is not None:
        return hparams["bias"]
    bias = non_quantized.get(f"{name}.bias")
    if bias is not None:
        return bias
    return None


def _load_or_normalize_qparams(
    qparams: Mapping[str, Mapping[str, Any]] | str | Path | None,
) -> dict[str, dict[str, Any]]:
    if qparams is None:
        return {"__default__": default_activation_qparams()}
    if isinstance(qparams, (str, Path)):
        return load_activation_qparams(qparams)
    return {str(key): _normalize_act_qparams(value) for key, value in qparams.items()}


def _select_qparams(name: str, qparams_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if name in qparams_map:
        return _normalize_act_qparams(qparams_map[name])
    if "__default__" in qparams_map:
        return _normalize_act_qparams(qparams_map["__default__"])
    raise KeyError(f"missing activation qparams for layer {name!r}")


def _normalize_act_qparams(value: Mapping[str, Any]) -> dict[str, Any]:
    if "scale" not in value or "zero_point" not in value:
        raise ValueError("activation qparams must include scale and zero_point")
    scale = float(value["scale"])
    if scale <= 0.0:
        raise ValueError("activation scale must be positive")
    zero_point = int(value["zero_point"])
    qmin = int(value.get("qmin", QMIN_U4))
    qmax = int(value.get("qmax", QMAX_U4))
    if qmin != QMIN_U4 or qmax != QMAX_U4:
        raise ValueError("this backend currently supports only u4 activation range [0, 15]")
    if zero_point < qmin or zero_point > qmax:
        raise ValueError("activation zero_point must be in [0, 15]")
    out = dict(value)
    out.update(
        {
            "bit_a": 4,
            "scale": scale,
            "zero_point": zero_point,
            "qmin": qmin,
            "qmax": qmax,
            "granularity": value.get("granularity", "per_tensor"),
        }
    )
    return out

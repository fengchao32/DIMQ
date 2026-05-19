"""Export helpers for DIMQ-quantized models."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .dimq_wrapper import DIMQConv2d, DIMQLinear, DIMQModule, get_dimq_modules


@torch.no_grad()
def convert_to_dequantized_model(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Return a model whose DIMQ layers are replaced by float Conv/Linear layers.

    The exported weights are nearest-center dequantized tensors, so each
    quantized layer contains at most ``2 ** w_bits`` unique float values.
    """

    if not inplace:
        model = copy.deepcopy(model)

    for name, module in list(get_dimq_modules(model)):
        parent, leaf = _resolve_parent(model, name)
        parent._modules[leaf] = module.to_float_module(quantize_weight=True)
    return model


@torch.no_grad()
def export_dequantized_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a dequantized state dict that normal PyTorch models can load."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dequantized = convert_to_dequantized_model(model, inplace=False)
    state = {
        "format": "dimq_dequantized",
        "model_state_dict": dequantized.state_dict(),
        "quantized_layers": [name for name, _ in get_dimq_modules(model)],
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    return state


@torch.no_grad()
def export_compact_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
    include_model_state: bool = False,
) -> dict[str, Any]:
    """Save DIMQ codebooks and nearest-center indices for every wrapped layer."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dimq_modules = get_dimq_modules(model)
    layers: dict[str, dict[str, Any]] = {}
    for name, module in dimq_modules:
        centers = module.centers.detach().float().sort().values
        weight, indices = _hard_quantize_with_sorted_centers(module, centers)
        index_dtype = torch.uint8 if centers.numel() <= 256 else torch.int16
        layers[name] = {
            "shape": list(module.weight.shape),
            "bits": int(module.cfg.w_bits),
            "centers": centers.cpu(),
            "indices": indices.detach().cpu().to(index_dtype),
            "dequantized_weight_dtype": str(weight.dtype).replace("torch.", ""),
        }
        if module.cfg.a_bits is not None:
            layers[name]["activation_bits"] = int(module.cfg.a_bits)
            layers[name]["activation_centers"] = module.activation_centers.detach().float().cpu()
            layers[name]["activation_centers_initialized"] = bool(
                module.activation_centers_initialized.detach().cpu().item()
            )

    first_cfg = None
    if dimq_modules:
        first_cfg = dimq_modules[0][1].cfg.to_dict()

    state: dict[str, Any] = {
        "format": "dimq_compact",
        "model_meta": {
            "num_quantized_layers": len(dimq_modules),
            "quantized_layer_names": [name for name, _ in dimq_modules],
        },
        "quant_cfg": first_cfg,
        "layers": layers,
        "non_quantized_state_dict": _non_quantized_state_dict(model, dimq_modules),
    }
    if include_model_state:
        state["model_state_dict"] = model.state_dict()
    if extra:
        state["extra"] = extra

    torch.save(state, path)
    return state


def load_compact_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


@torch.no_grad()
def assert_unique_values_within_codebook(model: nn.Module) -> None:
    """Raise if any DIMQ layer hard-quantizes to more values than its codebook."""

    for name, module in get_dimq_modules(model):
        weight, _ = module.hard_quantized_weight()
        unique_count = torch.unique(weight.detach().float()).numel()
        if unique_count > module.K:
            raise AssertionError(f"{name} has {unique_count} unique values, expected <= {module.K}")


@torch.no_grad()
def _hard_quantize_with_sorted_centers(
    module: DIMQModule,
    sorted_centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original = module.centers
    from .dimq import hard_quantize_weight

    return hard_quantize_weight(module.weight, sorted_centers.to(device=original.device, dtype=original.dtype), module.cfg.chunk_size)


def _resolve_parent(model: nn.Module, layer_name: str) -> tuple[nn.Module, str]:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent._modules[part]
    return parent, parts[-1]


def _non_quantized_state_dict(
    model: nn.Module,
    dimq_modules: list[tuple[str, DIMQModule]],
) -> dict[str, torch.Tensor]:
    skip_keys = set()
    for name, _ in dimq_modules:
        skip_keys.add(f"{name}.weight")
        skip_keys.add(f"{name}.centers")
        skip_keys.add(f"{name}.activation_centers")
        skip_keys.add(f"{name}.activation_centers_initialized")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key not in skip_keys
    }

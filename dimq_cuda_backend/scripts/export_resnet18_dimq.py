#!/usr/bin/env python
"""Export the repository's DIMQ ResNet18 compact checkpoint to packed W4A4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dimq_backend.export import (
    export_packed_dimq_checkpoint,
    infer_module_hparams,
    load_activation_qparams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact",
        type=Path,
        default=Path(
            "/home/fengchao/DIMQ/checkpoints/dimq_resnet18_w4_lam5e6_tau01_sep1e2_clr2/best_dimq_compact.pth"
        ),
        help="DIMQ compact checkpoint produced by quant/export_dimq.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/fengchao/DIMQ/checkpoints/dimq_resnet18_w4_lam5e6_tau01_sep1e2_clr2/best_dimq_packed_w4a4.pt"
        ),
        help="Output packed W4A4 checkpoint path",
    )
    parser.add_argument(
        "--activation-qparams",
        type=Path,
        default=None,
        help="Optional JSON mapping layer name to {'scale': float, 'zero_point': int}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from torchvision.models import resnet18
    except Exception as exc:
        raise RuntimeError("torchvision is required to infer ResNet18 Conv2d strides/padding") from exc

    model = resnet18(weights=None)
    layer_hparams = infer_module_hparams(model)
    act_qparams = load_activation_qparams(args.activation_qparams) if args.activation_qparams else None
    state = export_packed_dimq_checkpoint(
        args.compact,
        args.output,
        act_qparams=act_qparams,
        layer_hparams=layer_hparams,
        model_name="resnet18_dimq_w4a4",
        codebook_dtype=torch.float16,
    )
    print(f"wrote {args.output}")
    print(f"layers: {len(state['layers'])}")
    print("activation qparams source:", "json" if act_qparams else "default_uncalibrated")


if __name__ == "__main__":
    main()

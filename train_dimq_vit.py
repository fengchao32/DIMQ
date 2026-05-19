"""Train DIMQ weight quantization on torchvision ViT-B/16.

The model is initialized from torchvision ImageNet pretrained weights before
DIMQ wraps Conv2d/Linear layers. For ViT-B/16 this covers the patch embedding
Conv2d and Transformer Linear projections, while the first and last eligible
layers are skipped by default.
"""

from __future__ import annotations

from train_dimq_resnet import main


if __name__ == "__main__":
    main(
        default_arch="vit_b_16",
        default_output_dir="checkpoints/dimq_vit_b_16",
        default_batch_size=128,
    )

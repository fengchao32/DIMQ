# DIMQ

DIMQ is a PyTorch implementation of differentiable image model quantization utilities for Conv2d and Linear layers. The repository includes quantization wrappers, export helpers, ResNet/ViT training entry points, and unit tests for the core math and export behavior.

## Contents

- `quant/dimq.py`: core DIMQ codebook initialization, annealed softmin loss, hard quantization, and statistics.
- `quant/dimq_wrapper.py`: PyTorch module wrappers for Conv2d and Linear layers.
- `quant/export_dimq.py`: dequantized and compact checkpoint export helpers.
- `train_dimq_resnet.py`: torchvision classifier training entry point.
- `train_dimq_vit.py`: ViT-B/16 training entry point using the shared trainer.
- `models/`: model-specific helpers.
- `tests/`: pytest coverage for math, wrappers, and export.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -q
```

## Training

Update the ImageNet paths in `dataset1k.py` for your local machine, then run one of the training entry points:

```bash
python train_dimq_resnet.py --arch resnet50 --epochs 80
python train_dimq_vit.py --epochs 80
```

Training checkpoints and exported artifacts are written under `checkpoints/` by default and are ignored by Git.

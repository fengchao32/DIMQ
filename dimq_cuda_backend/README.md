# DIMQ CUDA Backend

This package implements the packed-index DIMQ inference path described in
`../DIMQ_CUDA_CUTLASS_Implementation_Guide.md`.

The runtime representation is:

- weights: packed unsigned 4-bit nearest-center indices, two indices per byte
- weight codebook: learned non-uniform 16-entry codebook per layer
- activations: uniform affine unsigned 4-bit quantization
- compute: fused product-table lookup in CUDA kernels with FP32 accumulation

The full dequantized weight tensor is not materialized in global memory by the
CUDA kernels.  The included PyTorch reference can materialize it for correctness
checks only.

## Build

```bash
cd /home/fengchao/DIMQ/dimq_cuda_backend
python setup.py build_ext --inplace
```

or:

```bash
pip install -e /home/fengchao/DIMQ/dimq_cuda_backend
```

## Export The Provided ResNet18 W4 Checkpoint

```bash
python scripts/export_resnet18_dimq.py \
  --compact /home/fengchao/DIMQ/checkpoints/dimq_resnet18_w4/best_dimq_compact.pth \
  --output /home/fengchao/DIMQ/checkpoints/dimq_resnet18_w4/best_dimq_packed_w4a4.pt
```

For real accuracy runs, pass calibrated activation qparams:

```bash
python scripts/export_resnet18_dimq.py \
  --compact .../best_dimq_compact.pth \
  --output .../best_dimq_packed_w4a4.pt \
  --activation-qparams activation_qparams.json
```



## Operators

The extension registers these CUDA operators under `torch.ops.dimq`:

- `pack_u4`
- `unpack_u4`
- `quantize_activation_u4`
- `build_product_table`
- `linear_lut`
- `conv2d_lut`

The kernels are correctness-first CUDA SIMT kernels.  The ABI is intentionally
compatible with later CUTLASS-style tiled kernels.

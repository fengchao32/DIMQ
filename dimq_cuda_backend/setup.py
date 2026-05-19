from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent


setup(
    name="dimq-cuda-backend",
    version="0.1.0",
    description="Packed-index DIMQ W4A4 CUDA LUT inference backend",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="dimq_backend._C",
            sources=[
                str(ROOT / "csrc" / "dimq_ops.cpp"),
                str(ROOT / "csrc" / "dimq_pack.cu"),
                str(ROOT / "csrc" / "dimq_product_table.cu"),
                str(ROOT / "csrc" / "dimq_gemm_lut.cu"),
                str(ROOT / "csrc" / "dimq_conv2d_lut.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

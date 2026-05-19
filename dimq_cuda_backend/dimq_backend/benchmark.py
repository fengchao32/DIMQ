"""Small CUDA event benchmark helpers for DIMQ kernels."""

from __future__ import annotations

from collections.abc import Callable

import torch


def benchmark_cuda(fn: Callable[[], object], *, warmup: int = 50, repeat: int = 200) -> float:
    """Return average latency in milliseconds using CUDA events."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_cuda")
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(repeat)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(1, int(repeat))

from __future__ import annotations

import sys
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture()
def torch_mod():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


@pytest.fixture()
def cuda_backend(torch_mod):
    from dimq_backend import ops

    if not ops.is_available():
        pytest.skip("dimq_backend CUDA extension has not been built")
    return ops


def test_cuda_pack_unpack_matches_python(torch_mod, cuda_backend):
    from dimq_backend.export import pack_u4_indices

    indices = (torch_mod.arange(257, dtype=torch_mod.uint8, device="cuda") * 7) % 16
    packed_cuda = cuda_backend.pack_u4(indices)
    restored = cuda_backend.unpack_u4(packed_cuda, indices.numel())
    packed_py = pack_u4_indices(indices.cpu()).cuda()
    torch_mod.testing.assert_close(packed_cuda, packed_py)
    torch_mod.testing.assert_close(restored, indices)


def test_cuda_product_table_matches_reference(torch_mod, cuda_backend):
    from dimq_backend.reference import build_product_table

    codebook = torch_mod.linspace(-1.0, 1.0, 16, device="cuda", dtype=torch_mod.float16)
    table = cuda_backend.build_product_table(codebook, 0.125, 8)
    expected = build_product_table(codebook, 0.125, 8)
    torch_mod.testing.assert_close(table, expected, rtol=1e-3, atol=1e-4)


def test_cuda_linear_lut_matches_reference(torch_mod, cuda_backend):
    from dimq_backend.export import pack_u4_indices
    from dimq_backend.reference import dimq_linear_reference

    torch_mod.manual_seed(3)
    x = torch_mod.randn(4, 11, device="cuda")
    codebook = torch_mod.linspace(-0.5, 0.5, 16, device="cuda")
    indices = torch_mod.randint(0, 16, (6, 11), dtype=torch_mod.uint8)
    packed = pack_u4_indices(indices).cuda()
    bias = torch_mod.randn(6, device="cuda")

    y = cuda_backend.dimq_linear_lut(x, packed, codebook, 0.2, 7, bias, 6, 11)
    y_ref = dimq_linear_reference(x, packed.cpu(), codebook, 0.2, 7, 6, 11, bias)
    torch_mod.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)


def test_cuda_conv2d_lut_matches_reference(torch_mod, cuda_backend):
    from dimq_backend.export import pack_u4_indices
    from dimq_backend.reference import dimq_conv2d_reference

    torch_mod.manual_seed(4)
    x = torch_mod.randn(2, 3, 6, 7, device="cuda")
    codebook = torch_mod.linspace(-0.75, 0.75, 16, device="cuda")
    indices = torch_mod.randint(0, 16, (5, 3, 3, 3), dtype=torch_mod.uint8)
    packed = pack_u4_indices(indices).cuda()
    bias = torch_mod.randn(5, device="cuda")
    kwargs = dict(stride=(2, 1), padding=(1, 1), dilation=(1, 1), groups=1)

    y = cuda_backend.dimq_conv2d_lut(
        x,
        packed,
        codebook,
        0.25,
        7,
        bias,
        indices.shape,
        kwargs["stride"],
        kwargs["padding"],
        kwargs["dilation"],
        kwargs["groups"],
    )
    y_ref = dimq_conv2d_reference(x, packed.cpu(), codebook, 0.25, 7, indices.shape, bias, **kwargs)
    torch_mod.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)

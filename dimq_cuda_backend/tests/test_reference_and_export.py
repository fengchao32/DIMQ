from __future__ import annotations

import sys
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture()
def torch_mod():
    return pytest.importorskip("torch")


def test_pack_unpack_u4_round_trip_odd_length(torch_mod):
    from dimq_backend.export import pack_u4_indices, unpack_u4_indices

    indices = torch_mod.arange(17, dtype=torch_mod.uint8) % 16
    packed = pack_u4_indices(indices)
    assert packed.dtype == torch_mod.uint8
    assert packed.numel() == 9
    restored = unpack_u4_indices(packed, indices.numel())
    assert torch_mod.equal(restored, indices)


def test_linear_reference_matches_explicit_dequant(torch_mod):
    from dimq_backend.export import pack_u4_indices
    from dimq_backend.reference import dimq_linear_dequant_reference, dimq_linear_reference

    torch_mod.manual_seed(0)
    x = torch_mod.randn(3, 7)
    codebook = torch_mod.linspace(-1.0, 1.0, 16)
    indices = torch_mod.randint(0, 16, (5, 7), dtype=torch_mod.uint8)
    packed = pack_u4_indices(indices)
    bias = torch_mod.randn(5)

    y_lut = dimq_linear_reference(x, packed, codebook, 0.125, 8, 5, 7, bias)
    y_deq = dimq_linear_dequant_reference(x, packed, codebook, 0.125, 8, 5, 7, bias)
    torch_mod.testing.assert_close(y_lut, y_deq, rtol=0.0, atol=1e-6)


def test_conv_reference_matches_explicit_dequant(torch_mod):
    from dimq_backend.export import pack_u4_indices
    from dimq_backend.reference import dimq_conv2d_lut_reference, dimq_conv2d_reference

    torch_mod.manual_seed(1)
    x = torch_mod.randn(2, 3, 5, 6)
    codebook = torch_mod.linspace(-0.75, 0.75, 16)
    indices = torch_mod.randint(0, 16, (4, 3, 3, 3), dtype=torch_mod.uint8)
    packed = pack_u4_indices(indices)
    bias = torch_mod.randn(4)

    kwargs = dict(stride=(2, 1), padding=(1, 1), dilation=(1, 1), groups=1)
    y_lut = dimq_conv2d_lut_reference(x, packed, codebook, 0.25, 7, indices.shape, bias, **kwargs)
    y_deq = dimq_conv2d_reference(x, packed, codebook, 0.25, 7, indices.shape, bias, **kwargs)
    torch_mod.testing.assert_close(y_lut, y_deq, rtol=0.0, atol=1e-5)


def test_export_provided_resnet18_w4_checkpoint_smoke(tmp_path, torch_mod):
    from dimq_backend.export import export_packed_dimq_checkpoint

    checkpoint = Path(
        "/home/fengchao/DIMQ/checkpoints/dimq_resnet18_w4_lam5e6_tau01_sep1e2_clr2/best_dimq_compact.pth"
    )
    if not checkpoint.exists():
        pytest.skip("local ResNet18 W4 compact checkpoint is not present")

    out = tmp_path / "packed.pt"
    state = export_packed_dimq_checkpoint(checkpoint, out)
    assert out.exists()
    assert state["format"] == "dimq_packed_lut"
    assert state["activation_quantization"] == "uniform_affine_u4"
    assert len(state["layers"]) == 16
    first = state["layers"]["layer1.0.conv1"]
    assert first["bit_w"] == 4
    assert first["bit_a"] == 4
    assert first["codebook"].numel() == 16
    assert first["packed_weight"].dtype == torch_mod.uint8
    assert first["packed_weight"].numel() == (first["weight_numel"] + 1) // 2

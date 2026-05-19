import pytest


@pytest.fixture()
def dimq_export():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from quant import DIMQConfig, apply_dimq
    from quant.export_dimq import convert_to_dequantized_model, export_compact_checkpoint

    return {
        "torch": torch,
        "nn": nn,
        "DIMQConfig": DIMQConfig,
        "apply_dimq": apply_dimq,
        "convert_to_dequantized_model": convert_to_dequantized_model,
        "export_compact_checkpoint": export_compact_checkpoint,
    }


def test_dequantized_model_has_no_dimq_centers(dimq_export):
    torch = dimq_export["torch"]
    nn = dimq_export["nn"]
    DIMQConfig = dimq_export["DIMQConfig"]
    apply_dimq = dimq_export["apply_dimq"]
    convert_to_dequantized_model = dimq_export["convert_to_dequantized_model"]
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False))
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False, center_init="quantile")
    apply_dimq(model, cfg)

    exported = convert_to_dequantized_model(model, inplace=False)
    keys = exported.state_dict().keys()
    assert not any("centers" in key for key in keys)
    assert torch.unique(exported[0].weight.detach()).numel() <= 2 ** cfg.w_bits


def test_compact_checkpoint_contains_centers_and_indices(tmp_path, dimq_export):
    nn = dimq_export["nn"]
    DIMQConfig = dimq_export["DIMQConfig"]
    apply_dimq = dimq_export["apply_dimq"]
    export_compact_checkpoint = dimq_export["export_compact_checkpoint"]
    model = nn.Sequential(nn.Linear(8, 4, bias=False))
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False, center_init="quantile")
    apply_dimq(model, cfg)

    state = export_compact_checkpoint(model, tmp_path / "compact.pth", include_model_state=False)
    layer = state["layers"]["0"]
    assert layer["centers"].numel() == 2 ** cfg.w_bits
    assert layer["indices"].shape == model[0].weight.shape

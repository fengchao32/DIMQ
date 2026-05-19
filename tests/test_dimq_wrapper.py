import pytest


@pytest.fixture()
def dimq_wrapper():
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from quant import DIMQConfig, apply_dimq, get_dimq_modules
    from quant.dimq_wrapper import collect_quant_layers

    return {
        "torch": torch,
        "nn": nn,
        "DIMQConfig": DIMQConfig,
        "apply_dimq": apply_dimq,
        "collect_quant_layers": collect_quant_layers,
        "get_dimq_modules": get_dimq_modules,
    }


def test_apply_dimq_wraps_inner_resnet_like_layers(dimq_wrapper):
    torch = dimq_wrapper["torch"]
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    apply_dimq = dimq_wrapper["apply_dimq"]
    get_dimq_modules = dimq_wrapper["get_dimq_modules"]
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )
    cfg = DIMQConfig(w_bits=3, skip_first=True, skip_last=True, center_init="quantile")
    modules = apply_dimq(model, cfg)
    assert len(modules) == 1
    assert modules[0].centers.requires_grad
    assert len(get_dimq_modules(model)) == 1

    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)


def test_dimq_losses_are_finite(dimq_wrapper):
    torch = dimq_wrapper["torch"]
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    apply_dimq = dimq_wrapper["apply_dimq"]
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False))
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False, center_init="quantile")
    modules = apply_dimq(model, cfg)
    distortion, separation = modules[0].dimq_losses()
    assert torch.isfinite(distortion)
    assert torch.isfinite(separation)


def test_collect_quant_layers_skips_downsample_by_default(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    collect_quant_layers = dimq_wrapper["collect_quant_layers"]

    class TinyResNetLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
            self.downsample = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(4),
            )
            self.proj = nn.Conv2d(4, 4, kernel_size=1, bias=False)

    model = TinyResNetLike()
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False)
    names = [name for name, _ in collect_quant_layers(model, cfg)]
    assert names == ["conv", "proj"]

    cfg.skip_downsample = False
    names = [name for name, _ in collect_quant_layers(model, cfg)]
    assert names == ["conv", "downsample.0", "proj"]


def test_collect_quant_layers_includes_depthwise_conv(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    collect_quant_layers = dimq_wrapper["collect_quant_layers"]

    model = nn.Sequential(
        nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False),
        nn.Conv2d(8, 16, kernel_size=1, bias=False),
    )
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False)
    names = [name for name, _ in collect_quant_layers(model, cfg)]

    assert names == ["0", "1"]


def test_collect_quant_layers_can_skip_depthwise_conv(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    collect_quant_layers = dimq_wrapper["collect_quant_layers"]

    model = nn.Sequential(
        nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False),
        nn.Conv2d(8, 16, kernel_size=1, bias=False),
    )
    cfg = DIMQConfig(w_bits=2, skip_first=False, skip_last=False, skip_depthwise=True)
    names = [name for name, _ in collect_quant_layers(model, cfg)]

    assert names == ["1"]


def test_collect_quant_layers_supports_name_patterns(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    DIMQConfig = dimq_wrapper["DIMQConfig"]
    collect_quant_layers = dimq_wrapper["collect_quant_layers"]

    class TinyDetectorLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
                nn.Linear(8, 8),
            )
            self.roi_heads = nn.Sequential(
                nn.Linear(8, 4),
            )

    model = TinyDetectorLike()
    cfg = DIMQConfig(
        w_bits=2,
        skip_first=False,
        skip_last=False,
        include_patterns=("backbone.*",),
        exclude_patterns=("*.1",),
    )
    names = [name for name, _ in collect_quant_layers(model, cfg)]

    assert names == ["backbone.0"]


def test_replace_classifier_head_supports_mobilenetv2_style_classifier(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    from train_dimq_resnet import replace_classifier_head

    class TinyMobileNetV2Like(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1280, 1000),
            )

    model = TinyMobileNetV2Like()
    replace_classifier_head(model, "mobilenet_v2", 100)

    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].in_features == 1280
    assert model.classifier[1].out_features == 100


def test_replace_classifier_head_supports_vit_style_heads(dimq_wrapper):
    nn = dimq_wrapper["nn"]
    from train_dimq_resnet import replace_classifier_head

    class TinyViTLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 1000),
            )

    model = TinyViTLike()
    replace_classifier_head(model, "vit_b_16", 100)

    assert isinstance(model.heads[1], nn.Linear)
    assert model.heads[1].in_features == 768
    assert model.heads[1].out_features == 100

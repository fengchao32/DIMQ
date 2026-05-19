import pytest


@pytest.fixture()
def dimq_math():
    torch = pytest.importorskip("torch")
    from quant.dimq import (
        DIMQConfig,
        dimq_softmin_loss,
        get_tau,
        hard_quantize_weight,
        separation_loss,
        soft_assignment_stats,
    )

    return {
        "torch": torch,
        "DIMQConfig": DIMQConfig,
        "dimq_softmin_loss": dimq_softmin_loss,
        "get_tau": get_tau,
        "hard_quantize_weight": hard_quantize_weight,
        "separation_loss": separation_loss,
        "soft_assignment_stats": soft_assignment_stats,
    }


def test_tau_exponential_reaches_endpoints(dimq_math):
    DIMQConfig = dimq_math["DIMQConfig"]
    get_tau = dimq_math["get_tau"]
    cfg = DIMQConfig(tau_start=1.0, tau_end=1e-5, tau_schedule="exponential")
    assert get_tau(0.0, cfg) == pytest.approx(1.0)
    assert get_tau(1.0, cfg) == pytest.approx(1e-5)


def test_hard_quantize_unique_values_within_codebook(dimq_math):
    torch = dimq_math["torch"]
    hard_quantize_weight = dimq_math["hard_quantize_weight"]
    w = torch.linspace(-2.0, 2.0, 101)
    centers = torch.tensor([-1.0, 0.0, 1.0])
    w_q, idx = hard_quantize_weight(w, centers, chunk_size=17)
    assert w_q.shape == w.shape
    assert idx.shape == w.shape
    assert torch.unique(w_q).numel() <= centers.numel()


def test_assignment_entropy_decreases_with_tau(dimq_math):
    torch = dimq_math["torch"]
    soft_assignment_stats = dimq_math["soft_assignment_stats"]
    w = torch.linspace(-1.0, 1.0, 128)
    centers = torch.tensor([-1.0, -0.25, 0.25, 1.0])
    high_tau = soft_assignment_stats(w, centers, tau=1.0, chunk_size=32)
    low_tau = soft_assignment_stats(w, centers, tau=1e-4, chunk_size=32)
    assert low_tau["avg_assignment_entropy"] < high_tau["avg_assignment_entropy"]
    assert torch.isclose(low_tau["hard_codebook_usage"].sum(), torch.tensor(1.0))


def test_separation_loss_zero_when_centers_exceed_margin(dimq_math):
    torch = dimq_math["torch"]
    separation_loss = dimq_math["separation_loss"]
    w = torch.randn(256)
    centers = torch.tensor([-10.0, -5.0, 0.0, 5.0])
    loss = separation_loss(w, centers, eta=1.0)
    assert loss.item() == pytest.approx(0.0)


def test_softmin_loss_is_finite(dimq_math):
    torch = dimq_math["torch"]
    dimq_softmin_loss = dimq_math["dimq_softmin_loss"]
    w = torch.randn(513)
    centers = torch.linspace(-1.0, 1.0, 8)
    loss = dimq_softmin_loss(w, centers, tau=1e-5, chunk_size=64)
    assert torch.isfinite(loss)

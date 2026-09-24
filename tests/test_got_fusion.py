import pytest
import torch

from swingot.model import CrossModalGOTFusion


def test_got_fusion_forward_losses_and_gradients():
    torch.manual_seed(0)
    fusion = CrossModalGOTFusion(
        latent_dim=8,
        hidden_dim=16,
        dropout=0.0,
        sinkhorn_iterations=10,
        max_samples=16,
    )
    first = torch.randn(20, 8, requires_grad=True)
    second = first.detach() + 0.1 * torch.randn(20, 8)
    second.requires_grad_(True)

    fused, gate = fusion(first, second, return_gate=True)
    losses = fusion.alignment_losses(first, second)
    total = fused.square().mean() + sum(losses.values())
    total.backward()

    assert fused.shape == first.shape
    assert gate.shape == (20, 2)
    assert torch.allclose(gate.sum(dim=1), torch.ones(20), atol=1e-6)
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(value.item() >= 0.0 for value in losses.values())
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()
    assert any(parameter.grad is not None for parameter in fusion.parameters())


def test_got_fusion_rejects_unpaired_forward():
    fusion = CrossModalGOTFusion(latent_dim=4, dropout=0.0)
    with pytest.raises(ValueError, match="paired modalities"):
        fusion(torch.randn(3, 4), torch.randn(4, 4))

"""Model architecture regression tests."""

import pytest


torch = pytest.importorskip("torch")

from molu_chatbot import CausalSelfAttention


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_causal_self_attention_forward_shape_and_finiteness(device: str) -> None:
    torch.manual_seed(42)
    attn = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    attn.eval().to(device)

    x = torch.randn(2, 6, 32, device=device)
    mask = torch.ones(2, 6, device=device)
    mask[0, -1] = 0

    with torch.no_grad():
        output = attn(x, attn_mask=mask)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_causal_self_attention_cpu_cuda_consistency() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for consistency check")

    torch.manual_seed(1234)
    attn_cpu = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.0).eval()
    x_cpu = torch.randn(2, 6, 32)
    mask_cpu = torch.ones(2, 6)
    mask_cpu[0, -1] = 0

    with torch.no_grad():
        cpu_out = attn_cpu(x_cpu, attn_mask=mask_cpu)

    attn_cuda = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.0).eval().cuda()
    attn_cuda.load_state_dict(attn_cpu.state_dict())

    with torch.no_grad():
        cuda_out = attn_cuda(x_cpu.cuda(), attn_mask=mask_cpu.cuda()).cpu()

    assert torch.allclose(cpu_out, cuda_out, atol=1e-4, rtol=1e-3)

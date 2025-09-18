import math

import pytest

torch = pytest.importorskip("torch")

from molu_chatbot import CausalSelfAttention, GPTScratch


def test_gpt_weight_tying_and_forward():
    vocab_size = 32
    model = GPTScratch(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_len=32,
        dropout=0.1,
    )

    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()

    tokens = torch.randint(0, vocab_size, (2, 16))
    output = model(tokens)

    assert output.shape == (2, 16, vocab_size)


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_causal_self_attention_matches_reference(device: str) -> None:
    torch.manual_seed(0)
    attn = CausalSelfAttention(d_model=32, n_heads=4, dropout=0.1).to(device)
    attn.eval()

    x = torch.randn(2, 6, 32, device=device, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(2, 6, device=device, dtype=torch.long)
    mask[0, -1] = 0

    with torch.no_grad():
        output_sdpa = attn(x, mask)

    # Reference implementation mirroring the previous attention logic.
    with torch.no_grad():
        qkv = attn.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(2, 6, attn.nh, attn.dk).transpose(1, 2)
        k = k.view(2, 6, attn.nh, attn.dk).transpose(1, 2)
        v = v.view(2, 6, attn.nh, attn.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.dk)
        causal_mask = torch.tril(
            torch.ones(6, 6, device=device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        manual = probs @ v
        manual = manual.transpose(1, 2).contiguous().view(2, 6, 32)
        manual = attn.resid_drop(attn.proj(manual))

    assert output_sdpa.shape == manual.shape == (2, 6, 32)
    assert torch.allclose(output_sdpa, manual, atol=1e-5, rtol=1e-4)

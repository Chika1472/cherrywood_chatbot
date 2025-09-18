import pytest

torch = pytest.importorskip("torch")

from molu_chatbot import GPTScratch


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

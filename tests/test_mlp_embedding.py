import torch
from model.mlp import AdvancedMLP


def test_advancedmlp_embedding_forward():
    vocab_size = 10
    embed_dim = 16
    model = AdvancedMLP(input_size=embed_dim, hidden_layers=[8], output_size=vocab_size,
                        use_embedding=True, vocab_size=vocab_size, embedding_dim=embed_dim)
    # Pass an index tensor (batch size 2 to satisfy BatchNorm when training)
    idx = torch.tensor([3, 4], dtype=torch.long)
    out = model(idx)
    # Should produce logits over vocab
    assert out.shape[-1] == vocab_size
    assert out.dtype == torch.float32

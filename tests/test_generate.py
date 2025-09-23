import torch
import torch.nn as nn
from inference.generate import generate_text_simple


def test_generate_lstm_basic():
    # tiny vocab
    word_to_index = {'<PAD>':0, '<UNK>':1, 'a':2, 'b':3}
    index_to_word = {v:k for k,v in word_to_index.items()}

    class DummyLSTM(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        def init_hidden(self, batch_size, device):
            return torch.zeros(1, batch_size, 1)
        def forward(self, x, hidden):
            batch = x.size(0)
            logits = torch.ones(batch, self.vocab_size, dtype=torch.float)
            return logits, None, hidden

    model = DummyLSTM(vocab_size=len(word_to_index))
    out = generate_text_simple(model, ['a','b'], word_to_index, index_to_word, num_words=5, temperature=1.0, device='cpu', model_type='lstm')
    assert isinstance(out, str)
    assert len(out) > 0
    assert '<PAD>' not in out and '<UNK>' not in out


def test_generate_mlp_basic():
    word_to_index = {'<PAD>':0, '<UNK>':1, 'x':2}
    index_to_word = {v:k for k,v in word_to_index.items()}
    class DummyMLP(nn.Module):
        def __init__(self, vocab_size, embed_dim=8):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.out = nn.Linear(embed_dim, vocab_size)
        def forward(self, x):
            # x may be Long indices (batch,) when using embed
            if x.dtype == torch.long:
                emb = self.embed(x)
            else:
                emb = x
            return self.out(emb)

    model = DummyMLP(vocab_size=len(word_to_index))
    out = generate_text_simple(model, ['x'], word_to_index, index_to_word, num_words=3, temperature=1.0, device='cpu', model_type='mlp')
    assert isinstance(out, str)
    assert len(out) > 0
    assert '<PAD>' not in out and '<UNK>' not in out

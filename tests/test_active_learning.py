import torch
import torch.nn as nn

from data import active_learning as al

class DummySeqModel(nn.Module):
    def __init__(self, vocab_size=11, seq_return=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_return = seq_return
        self.embed = nn.Embedding(vocab_size, 8)
        self.rnn = nn.GRU(8, 16, batch_first=True)
        self.head = nn.Linear(16, vocab_size)
    def init_hidden(self, batch, device):
        return torch.zeros(1, batch, 16, device=device)
    def forward(self, x, hidden=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        emb = self.embed(x)
        out, h = self.rnn(emb, hidden if hidden is not None else self.init_hidden(x.size(0), emb.device))
        logits = self.head(out)
        if not self.seq_return:  # collapse to (B,V)
            logits = logits[:, -1, :]
        return logits, None, h


def test_entropy_selection_sequence():
    model = DummySeqModel()
    inputs = torch.randint(0, 11, (20, 12))
    sel = al.select_uncertain_samples(model, inputs, top_k=5, device='cpu', strategy='entropy')
    assert sel.shape[0] == 5


def test_margin_selection_sequence():
    model = DummySeqModel()
    inputs = torch.randint(0, 11, (15, 10))
    sel = al.select_uncertain_samples(model, inputs, top_k=7, device='cpu', strategy='margin')
    assert sel.shape[0] == 7


def test_variation_ratio_sequence():
    model = DummySeqModel()
    inputs = torch.randint(0, 11, (12, 9))
    sel = al.select_uncertain_samples(model, inputs, top_k=4, device='cpu', strategy='variation_ratio', mc_passes=3)
    assert sel.shape[0] == 4


def test_handles_single_sample():
    model = DummySeqModel()
    inputs = torch.randint(0, 11, (9,))  # 1D
    sel = al.select_uncertain_samples(model, inputs, top_k=1, device='cpu', strategy='entropy')
    assert sel.shape[0] == 1


def test_top_k_clip():
    model = DummySeqModel()
    inputs = torch.randint(0, 11, (3, 6))
    sel = al.select_uncertain_samples(model, inputs, top_k=10, device='cpu', strategy='entropy')
    assert sel.shape[0] == 3  # clipped

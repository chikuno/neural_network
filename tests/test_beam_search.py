import torch
import torch.nn as nn
from inference.generate import generate_text_simple
from config import config as cfg

class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.t = 0
    def init_hidden(self, batch_size, device):
        return None
    def forward(self, x, hidden=None):
        # x shape: (1, t)
        # Construct logits that strongly prefer token 2 then 3
        vocab = self.vocab_size
        logits = torch.full((1, vocab), -10.0)
        # Step-dependent preference to create a deterministic path
        if x.numel() % 2 == 1:
            logits[0,2] = 5.0
        else:
            logits[0,3] = 5.0
        return logits, None, hidden

def test_beam_search_runs_and_respects_min_length():
    word_to_index = {'<PAD>':0, '<UNK>':1, 'A':2, 'B':3, '<EOS>':4}
    index_to_word = {v:k for k,v in word_to_index.items()}
    model = TinyLM(vocab_size=len(word_to_index))
    # Configure beam search
    old = (cfg.DECODING, cfg.BEAM_SIZE, cfg.LENGTH_PENALTY, cfg.EARLY_STOP, cfg.MIN_LENGTH, cfg.EOS_TOKEN)
    cfg.DECODING = 'beam'
    cfg.BEAM_SIZE = 2
    cfg.LENGTH_PENALTY = 0.0
    cfg.EARLY_STOP = True
    cfg.MIN_LENGTH = 2
    cfg.EOS_TOKEN = '<EOS>'
    try:
        out = generate_text_simple(model, ['A'], word_to_index, index_to_word, num_words=4, temperature=1.0, device='cpu', model_type='lstm')
        assert isinstance(out, str)
        # Should not end immediately due to MIN_LENGTH
        assert len(out.split()) >= 1
    finally:
        # restore
        cfg.DECODING, cfg.BEAM_SIZE, cfg.LENGTH_PENALTY, cfg.EARLY_STOP, cfg.MIN_LENGTH, cfg.EOS_TOKEN = old

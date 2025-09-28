import os
import json
import torch
from config import config as cfg
from inference.generate import generate_text_learned_ensemble

class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.lin = torch.nn.Linear(8, vocab_size)
    def init_hidden(self, b, device):
        return None
    def forward(self, x, h=None, src_key_padding_mask=None):
        emb = self.embed(x)
        out = self.lin(emb)
        return out, None


def test_dynamic_ensemble_weights(tmp_path):
    # Setup fake vocab
    vocab = {'<UNK>':0,'alpha':1,'beta':2,'gamma':3,'<eos>':4}
    idx2 = {i:w for w,i in vocab.items()}
    # Fake stats: LSTM better (lower loss)
    stats = {
        'models': {
            'lstm': {'val_loss': 1.2, 'val_ppl': 3.3, 'step_count': 10},
            'transformer': {'val_loss': 2.4, 'val_ppl': 11.0, 'step_count': 8}
        }
    }
    with open('ensemble_stats.json','w',encoding='utf-8') as f:
        json.dump(stats,f)

    cfg.ENSEMBLE_MODE = 'learned_dynamic'
    cfg.GENERATION_TRACE = True
    cfg.TRACE_FILENAME = 'test_dynamic_trace.json'

    lstm = DummyModel(len(vocab))
    transformer = DummyModel(len(vocab))
    device = torch.device('cpu')
    text = generate_text_learned_ensemble({'lstm': lstm, 'transformer': transformer}, 'alpha beta', vocab, idx2, 5, 1.0, device)
    assert isinstance(text, str)
    # Inspect trace file for weights
    assert os.path.exists(cfg.TRACE_FILENAME)
    with open(cfg.TRACE_FILENAME,'r',encoding='utf-8') as f:
        trace = json.load(f)
    summary = trace[-1]['summary'] if isinstance(trace, list) else trace.get('summary', {})
    # We expect lstm_weight > transformer_weight due to lower loss
    lw = summary.get('lstm_weight')
    tw = summary.get('transformer_weight')
    assert lw is not None and tw is not None and lw > tw, 'Dynamic weighting did not favor lower-loss model.'

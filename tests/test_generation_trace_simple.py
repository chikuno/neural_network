import os
import json
import torch

# Minimal smoke test for generation trace.
# Assumes config.GENERATION_TRACE can be toggled and a simple model exists (lstm).

def test_generation_trace_simple():
    from config import config as cfg
    # Force trace on
    cfg.GENERATION_TRACE = True
    cfg.TRACE_FILENAME = 'test_generation_trace.json'
    # Import generation utilities
    from inference.generate import generate_text_simple

    # Build a tiny dummy vocabulary + dummy model with required interface
    vocab = {'<UNK>':0,'the':1,'cat':2,'sat':3,'on':4,'mat':5,'<eos>':6}
    index_to_word = {i:w for w,i in vocab.items()}

    class DummyRNN(torch.nn.Module):
        def __init__(self,vocab_size):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size,8)
            self.lin = torch.nn.Linear(8, vocab_size)
        def init_hidden(self,batch,device):
            return None
        def forward(self,x,h):
            # x: (1,T)
            emb = self.embed(x)
            # simple mean pooling per step to shape (1,T,8)
            out = self.lin(emb)
            return out, None, None
    model = DummyRNN(len(vocab))
    device = torch.device('cpu')
    start = 'the cat'
    out = generate_text_simple(model, start, vocab, index_to_word, num_words=5, temperature=1.0, device=device, model_type='lstm')
    assert isinstance(out,str)
    assert os.path.exists(cfg.TRACE_FILENAME), 'Trace file not created'
    with open(cfg.TRACE_FILENAME,'r',encoding='utf-8') as f:
        trace = json.load(f)
    assert any('step' in r for r in trace), 'No step records in trace'
    assert any('summary' in r for r in trace), 'No summary record in trace'

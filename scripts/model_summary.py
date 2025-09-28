"""Lightweight model parameter summary.

Can be invoked as:
  python scripts/model_summary.py
from the project root OR from inside the scripts directory. We ensure the
project root is on sys.path so that 'from config import config' works even
if the working directory differs.
"""

import os
import sys

# Add project root (parent of this scripts directory) to sys.path if missing
_here = os.path.abspath(os.path.dirname(__file__))
_root = os.path.abspath(os.path.join(_here, os.pardir))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from config import config  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit("Failed to import config; ensure you run from project root. Original: " + str(e))

try:
    import model  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit("Failed to import model package. Original: " + str(e))
try:
    import data.data as data
except Exception:
    data = None
try:
    w2i, _ = data.try_load_vocab() if data else ({}, {})
except Exception:
    w2i = {}
vocab_size = len(w2i) if w2i else 500  # fallback default


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())

def summarize():
    print(f"Vocab size (loaded or fallback): {vocab_size}")
    def _get(name, default):
        return getattr(config, name, default)
    emb = _get('EMBEDDING_DIM', 256)
    hid = _get('HIDDEN_SIZE', 512)
    layers = _get('NUM_LAYERS', 2)
    try:
        hidden_layers = getattr(config, 'MODEL_CONFIG', {}).get('hidden_layers', [512,256])
    except Exception:
        hidden_layers = [512,256]
    models = {}
    try:
        models['rnn'] = model.RNNTextGenerationModel(vocab_size, emb, hid, layers, config.DROPOUT, config.MULTI_TASK)
        models['gru'] = model.GRUTextGenerationModel(vocab_size, emb, hid, layers, config.DROPOUT, config.MULTI_TASK)
        models['lstm'] = model.LSTMTextGenerationModel(vocab_size, emb, hid, layers, config.DROPOUT, config.MULTI_TASK)
    except Exception as e:
        print(f"Warning: failed to construct RNN family: {e}")
    try:
        models['transformer'] = model.TransformerTextGenerationModel(vocab_size, emb, config.NHEAD, layers, config.TRANSFORMER_DROPOUT, config.MULTI_TASK)
    except Exception as e:
        print(f"Warning: failed to construct transformer: {e}")
    try:
        models['mlp'] = model.MLPModel(vocab_size, emb, hidden_layers, vocab_size, config.DROPOUT)
    except Exception as e:
        print(f"Warning: failed to construct MLP: {e}")
    print("Model Parameter Summary (vocab_size may be fallback)")
    for name, m in models.items():
        print(f"{name.upper():12s}: {count_parameters(m):>10d} params")
        # Print core layer dims
        if name in ('rnn','gru','lstm'):
            print(f"  Embedding: {vocab_size} x {emb}")
            print(f"  Hidden size: {hid}, Layers: {layers}")
        elif name == 'transformer':
            ff_dim = 4*emb
            print(f"  Embedding: {vocab_size} x {emb}, Heads: {config.NHEAD}, Layers: {layers}, FF: {ff_dim}")
        elif name == 'mlp':
            try:
                print(f"  Hidden stack: {getattr(config, 'MODEL_CONFIG', {}).get('hidden_layers', hidden_layers)}")
            except Exception:
                pass
    print("\nNote: Re-run with actual vocab for precise counts.")

if __name__ == '__main__':
    summarize()

import os
import re
import torch
import gradio as gr

from config import config as cfg
import data.data as data
import model as model_pkg
# generation is now routed through ActiveRunner
from scripts.active_runner import get_runner


def _load_vocab():
    w2i, i2w = data.try_load_vocab()
    if not w2i or not i2w:
        # Build from current corpus as fallback
        txt = data.load_data(cfg.DATA_FILE)
        toks = data.preprocess_data(txt)
        w2i, i2w = data.build_vocabulary(toks, cfg.MIN_FREQUENCY)
    return w2i, i2w


def _detect_checkpoint(saved_dir='saved_models'):
    """Return (model_type, checkpoint_path) by scanning saved_models.

    Preference order: best_*.pth over last_*.pth; among types: transformer > lstm > gru > rnn.
    Fallback: (None, None) if nothing found.
    """
    if not os.path.isdir(saved_dir):
        return None, None
    files = [f for f in os.listdir(saved_dir) if f.endswith('.pth')]
    if not files:
        return None, None
    def parse(f):
        m = re.match(r"(best|last)_model_(rnn|lstm|gru|transformer)\.pth$", f)
        if not m:
            return None
        kind, mtype = m.group(1), m.group(2)
        return kind, mtype, f
    parsed = [p for p in map(parse, files) if p]
    if not parsed:
        return None, None
    # rank by kind then by type
    kind_rank = {'best': 0, 'last': 1}
    type_rank = {'transformer': 0, 'lstm': 1, 'gru': 2, 'rnn': 3}
    parsed.sort(key=lambda x: (kind_rank.get(x[0], 9), type_rank.get(x[1], 9)))
    kind, mtype, fname = parsed[0]
    return mtype, os.path.join(saved_dir, fname)

def _find_checkpoints(saved_dir='saved_models'):
    """Return mapping of model_type -> chosen checkpoint path (best over last).

    Types considered: 'transformer','lstm','gru','rnn'.
    """
    out = {k: None for k in ['transformer','lstm','gru','rnn']}
    if not os.path.isdir(saved_dir):
        return out
    files = [f for f in os.listdir(saved_dir) if f.endswith('.pth')]
    if not files:
        return out
    # Prefer best over last per type
    for t in out.keys():
        best = f"best_model_{t}.pth"
        last = f"last_model_{t}.pth"
        if best in files:
            out[t] = os.path.join(saved_dir, best)
        elif last in files:
            out[t] = os.path.join(saved_dir, last)
    return out

def _build_model(model_type, vocab_size, device):
    emb_dim = int(getattr(cfg, 'EMBEDDING_DIM', 128))
    hid = int(getattr(cfg, 'HIDDEN_SIZE', 256))
    layers = int(getattr(cfg, 'NUM_LAYERS', 2))
    dropout = float(getattr(cfg, 'DROPOUT', 0.5))
    multi_task = bool(getattr(cfg, 'MULTI_TASK', False))
    if model_type == 'transformer':
        nhead = int(getattr(cfg, 'NHEAD', 4))
        net = model_pkg.TransformerTextGenerationModel(vocab_size, emb_dim, nhead, layers, dropout, multi_task)
    elif model_type == 'gru':
        net = model_pkg.GRUTextGenerationModel(vocab_size, emb_dim, hid, layers, dropout, multi_task)
    elif model_type == 'rnn':
        net = model_pkg.RNNTextGenerationModel(vocab_size, emb_dim, hid, layers, dropout, multi_task)
    else:
        # default to lstm
        net = model_pkg.LSTMTextGenerationModel(vocab_size, emb_dim, hid, layers, dropout, multi_task)
    return net.to(device)

def _load_checkpoint_if_any(model, ckpt_path, device):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    try:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        return True
    except Exception as e:
        print('Warning: failed to load checkpoint:', e)
        return False

def _extract_state_dict(obj):
    """Return a state_dict-like mapping from a loaded checkpoint object."""
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        # assume obj itself is a state_dict
        return obj
    return None

def _expected_vocab_from_state(state_dict):
    """Infer vocab size from embedding or linear weights in the state dict."""
    if not isinstance(state_dict, dict):
        return None
    for key in ['embedding.weight', 'linear.weight']:
        if key in state_dict:
            w = state_dict[key]
            try:
                return int(w.shape[0])
            except Exception:
                try:
                    return int(w.size(0))
                except Exception:
                    pass
    # Try any *.weight with 2-D shape [vocab, dim] where vocab > dim
    try:
        for k, v in state_dict.items():
            if hasattr(v, 'dim') and v.dim() == 2:
                a, b = int(v.size(0)), int(v.size(1))
                if a > b and a > 10:
                    return a
    except Exception:
        pass
    return None

def _ensure_vocab_capacity(word_to_index, index_to_word, target_vocab):
    """Expand the local vocab mappings with placeholder tokens up to target size."""
    cur = len(word_to_index)
    if target_vocab is None or target_vocab <= cur:
        return word_to_index, index_to_word
    # Fill any missing ids from cur to target-1 with synthetic tokens
    for i in range(cur, target_vocab):
        tok = f"<tok_{i}>"
        word_to_index[tok] = i
        index_to_word[i] = tok
    return word_to_index, index_to_word

def _maybe_vocab_from_checkpoint(obj):
    """Extract vocab mappings from a checkpoint object if present.

    Returns (w2i, i2w) or (None, None).
    """
    if not isinstance(obj, dict):
        return None, None
    w2i = obj.get('word_to_index')
    i2w = obj.get('index_to_word')
    if not isinstance(w2i, dict) or not isinstance(i2w, (dict, list)):
        # Try nested keys
        for k in ['vocab','meta','extra']:
            sub = obj.get(k)
            if isinstance(sub, dict):
                w2i = sub.get('word_to_index', w2i)
                i2w = sub.get('index_to_word', i2w)
    if isinstance(i2w, dict):
        # Keys might be strings; coerce to int
        try:
            i2w = {int(k): v for k, v in i2w.items()}
        except Exception:
            pass
    if isinstance(w2i, dict) and i2w:
        return w2i, i2w
    return None, None


runner = get_runner()


with gr.Blocks(title="Neural Network Text Generator") as demo:
    header = "# Neural Network Text Generator\n"
    # Summarize available models/ckpts
    lines = []
    for t in ['transformer','lstm','gru','rnn']:
        ck = runner.ckpts.get(t) if hasattr(runner, 'ckpts') else None
        if ck:
            lines.append(f"{t.upper()}: {os.path.basename(ck)}")
    if lines:
        header += "Loaded checkpoints: " + ", ".join(lines)
    else:
        header += f"Using {runner.model_type.upper()} model (no checkpoint found; random init)"
    gr.Markdown(header)
    # Warning banner if placeholder vocab had to be synthesized
    if getattr(runner, 'using_placeholder_vocab', False) and getattr(runner, 'placeholder_tokens_added', 0) > 0:
        warn = (
            f"⚠️ Placeholder vocabulary in use (added {runner.placeholder_tokens_added} tokens).\n\n"
            "To get readable words instead of <tok_*>, restore the training-time vocabulary file at `data/vocab.json` "
            "or retrain; new checkpoints now embed vocab automatically."
        )
        gr.Markdown(warn)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Write a short story about a robot learning to dance…", lines=3)
    with gr.Row():
        num_tokens = gr.Slider(1, 200, value=int(getattr(cfg, 'NUM_WORDS_TO_GENERATE', 50)), step=1, label="Tokens to generate")
        temperature = gr.Slider(0.1, 2.0, value=float(getattr(cfg, 'TEMPERATURE', 1.0)), step=0.05, label="Temperature")
    with gr.Row():
        model_sel = gr.Dropdown(choices=["Auto","Auto (Best)","LSTM","Transformer","GRU","RNN","Ensemble"], value="Auto", label="Model")
        decoding = gr.Dropdown(choices=["sampling","beam"], value=str(getattr(cfg, 'DECODING','sampling')), label="Decoding")
        top_k = gr.Number(value=int(getattr(cfg,'TOP_K',0)), label="Top-K")
        top_p = gr.Number(value=float(getattr(cfg,'TOP_P',1.0)), label="Top-P")
        stream = gr.Checkbox(value=True, label="Stream output")

    out = gr.Textbox(label="Generated Text", lines=8)
    btn = gr.Button("Generate")

    def run_ui(prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel_v, stream):
        # UI path: allow streaming when checkbox is enabled using the active runner
        # Normalize selector to runner keywords
        if isinstance(model_sel_v, str) and model_sel_v.lower().startswith('auto') and 'best' in model_sel_v.lower():
            model_sel_v = 'auto_best'
        if stream:
            acc = []
            for chunk in runner.run_stream(prompt, num_tokens, temperature, decoding, top_k, top_p, model_choice=model_sel_v):
                acc.append(chunk)
                yield " ".join(acc)
            return " ".join(acc)
        return runner.run(prompt, num_tokens, temperature, decoding, top_k, top_p, model_choice=model_sel_v)

    btn.click(run_ui, inputs=[prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel, stream], outputs=out)

    # Non-streaming API endpoint for programmatic clients
    def run_api(prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel_v):
        if isinstance(model_sel_v, str) and model_sel_v.lower().startswith('auto') and 'best' in model_sel_v.lower():
            model_sel_v = 'auto_best'
        return runner.run(prompt, num_tokens, temperature, decoding, top_k, top_p, model_choice=model_sel_v)

    run_api_btn = gr.Button(visible=False)
    run_api_btn.click(run_api, inputs=[prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel], outputs=out, api_name="run")

    # Dedicated streaming endpoint: returns incremental text chunks; client may read as a single final string
    def run_stream(prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel_v):
        if isinstance(model_sel_v, str) and model_sel_v.lower().startswith('auto') and 'best' in model_sel_v.lower():
            model_sel_v = 'auto_best'
        acc = []
        for chunk in runner.run_stream(prompt, num_tokens, temperature, decoding, top_k, top_p, model_choice=model_sel_v):
            acc.append(chunk)
            # Yield partials for Gradio streaming UI (ignored by gradio_client predict)
            yield chunk
        # Final combined string
        return " ".join(acc)

    # Expose streaming API via a hidden route (using a hidden button to register the endpoint)
    stream_api = gr.Button(visible=False)
    stream_api.click(run_stream, inputs=[prompt, num_tokens, temperature, decoding, top_k, top_p, model_sel], outputs=out, api_name="run_stream")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860,)
    #share=True#

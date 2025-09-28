import os
import re
import json
import torch
from typing import Dict, Any, Generator

from config import config as cfg
import data.data as data
import model as model_pkg
from inference.generate import (
    generate_text_simple as _gen_simple,
    generate_stream as _gen_stream,
    generate_text_learned_ensemble as _gen_ensemble,
)

# ---- Helper utilities (mirrored from gradio_app) ----

def _detect_checkpoint(saved_dir: str = 'saved_models'):
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
    kind_rank = {'best': 0, 'last': 1}
    type_rank = {'transformer': 0, 'lstm': 1, 'gru': 2, 'rnn': 3}
    parsed.sort(key=lambda x: (kind_rank.get(x[0], 9), type_rank.get(x[1], 9)))
    kind, mtype, fname = parsed[0]
    return mtype, os.path.join(saved_dir, fname)

def _find_checkpoints(saved_dir: str = 'saved_models') -> Dict[str, str]:
    out = {k: None for k in ['transformer','lstm','gru','rnn']}
    if not os.path.isdir(saved_dir):
        return out
    files = [f for f in os.listdir(saved_dir) if f.endswith('.pth')]
    if not files:
        return out
    for t in out.keys():
        best = f"best_model_{t}.pth"
        last = f"last_model_{t}.pth"
        if best in files:
            out[t] = os.path.join(saved_dir, best)
        elif last in files:
            out[t] = os.path.join(saved_dir, last)
    return out

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        return obj
    return None

def _expected_vocab_from_state(state_dict):
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
    try:
        for _, v in state_dict.items():
            if hasattr(v, 'dim') and v.dim() == 2:
                a, b = int(v.size(0)), int(v.size(1))
                if a > b and a > 10:
                    return a
    except Exception:
        pass
    return None

def _ensure_vocab_capacity(word_to_index, index_to_word, target_vocab):
    cur = len(word_to_index)
    if target_vocab is None or target_vocab <= cur:
        return word_to_index, index_to_word
    for i in range(cur, target_vocab):
        tok = f"<tok_{i}>"
        word_to_index[tok] = i
        index_to_word[i] = tok
    return word_to_index, index_to_word

def _maybe_vocab_from_checkpoint(obj):
    if not isinstance(obj, dict):
        return None, None
    w2i = obj.get('word_to_index')
    i2w = obj.get('index_to_word')
    if not isinstance(w2i, dict) or not isinstance(i2w, (dict, list)):
        for k in ['vocab','meta','extra']:
            sub = obj.get(k)
            if isinstance(sub, dict):
                w2i = sub.get('word_to_index', w2i)
                i2w = sub.get('index_to_word', i2w)
    if isinstance(i2w, dict):
        try:
            i2w = {int(k): v for k, v in i2w.items()}
        except Exception:
            pass
    if isinstance(w2i, dict) and i2w:
        return w2i, i2w
    return None, None

def _build_model(model_type: str, vocab_size: int, device: torch.device):
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
        net = model_pkg.LSTMTextGenerationModel(vocab_size, emb_dim, hid, layers, dropout, multi_task)
    return net.to(device)

# ---- Active runner ----

class ActiveRunner:
    def __init__(self, device: str | None = None):
        self.device_str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_str)
        # Load vocab
        w2i, i2w = data.try_load_vocab()
        if not w2i or not i2w:
            txt = data.load_data(data.DATA_FILE)
            toks = data.preprocess_data(txt)
            w2i, i2w = data.build_vocabulary(toks, getattr(cfg, 'MIN_FREQUENCY', 1))
        self.word_to_index = w2i
        self.index_to_word = i2w
        # Checkpoints
        self.ckpts = _find_checkpoints()
        pref = ['transformer','lstm','gru','rnn']
        # Prefer best model by validation loss when available
        best = self._best_model_from_stats()
        if best and self.ckpts.get(best):
            self.model_type = best
        else:
            self.model_type = next((t for t in pref if self.ckpts.get(t)), 'lstm')
        self.models: Dict[str, Any] = {}
        self.loaded_ckpt = None
        self.placeholder_tokens_added = 0
        self.using_placeholder_vocab = False
        # Build primary
        self._build_and_load(self.model_type, self.ckpts.get(self.model_type))
        # Optionally pre-build partner models for ensemble availability
        if self.ckpts.get('transformer') and self.ckpts.get('lstm'):
            if self.model_type != 'transformer':
                self._build_and_load('transformer', self.ckpts.get('transformer'))
            if self.model_type != 'lstm':
                self._build_and_load('lstm', self.ckpts.get('lstm'))

    def _best_model_from_stats(self) -> str | None:
        """Pick the best model type using ensemble_stats.json val_loss across available checkpoints."""
        path = 'ensemble_stats.json'
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                stats = json.load(f) or {}
            models = stats.get('models', {}) or {}
            best_name = None
            best_loss = None
            for name in ['rnn','gru','lstm','transformer']:
                if not self.ckpts.get(name):
                    continue
                rec = models.get(name, {})
                vloss = rec.get('val_loss', None)
                if vloss is None:
                    continue
                try:
                    v = float(vloss)
                except Exception:
                    continue
                if v <= 0 or not (v == v):  # NaN/<=0 guard
                    continue
                if best_loss is None or v < best_loss:
                    best_loss = v
                    best_name = name
            return best_name
        except Exception:
            return None

    def _build_and_load(self, mtype: str, ckpt_path: str | None):
        vsize = len(self.word_to_index)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                blob = torch.load(ckpt_path, map_location=self.device)
                w2i_ck, i2w_ck = _maybe_vocab_from_checkpoint(blob)
                if isinstance(w2i_ck, dict) and isinstance(i2w_ck, (dict, list)):
                    self.word_to_index = dict(w2i_ck)
                    if isinstance(i2w_ck, list):
                        self.index_to_word = {i: t for i, t in enumerate(i2w_ck)}
                    else:
                        self.index_to_word = {int(k): v for k, v in i2w_ck.items()}
                    vsize = len(self.word_to_index)
                sdict = _extract_state_dict(blob)
                exp_vocab = _expected_vocab_from_state(sdict)
                if exp_vocab and exp_vocab != vsize:
                    before = len(self.word_to_index)
                    self.word_to_index, self.index_to_word = _ensure_vocab_capacity(self.word_to_index, self.index_to_word, exp_vocab)
                    added = len(self.word_to_index) - before
                    if added > 0 and not (isinstance(w2i_ck, dict) and isinstance(i2w_ck, (dict, list))):
                        self.placeholder_tokens_added += added
                        self.using_placeholder_vocab = True
                    vsize = exp_vocab
            except Exception:
                pass
        model = _build_model(mtype, vsize, self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=self.device)
                if isinstance(state, dict) and 'model' in state:
                    model.load_state_dict(state['model'], strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                self.loaded_ckpt = ckpt_path
            except Exception as e:
                print('Warning: failed to load checkpoint:', e)
        model.eval()
        self.models[mtype] = model

    # Synchronous generation (final string)
    def run(self, prompt: str, num_tokens: int, temperature: float, decoding: str, top_k: int, top_p: float, model_choice: str = "Auto") -> str:
        if not prompt:
            return ""
        toks = data.preprocess_data(prompt)
        start = " ".join(toks)
        # Temporarily override decoding settings
        old = {
            'DECODING': getattr(cfg, 'DECODING', 'sampling'),
            'TOP_K': getattr(cfg, 'TOP_K', 0),
            'TOP_P': getattr(cfg, 'TOP_P', 1.0),
            'EARLY_STOP': getattr(cfg, 'EARLY_STOP', True),
            'EOS_TOKEN': getattr(cfg, 'EOS_TOKEN', None),
            'MIN_LENGTH': getattr(cfg, 'MIN_LENGTH', 0),
        }
        cfg.DECODING = decoding
        cfg.TOP_K = int(top_k or 0)
        cfg.TOP_P = float(top_p or 1.0)
        if self.using_placeholder_vocab:
            try:
                cfg.EARLY_STOP = False
                cfg.EOS_TOKEN = None
                cfg.MIN_LENGTH = 0
            except Exception:
                pass
        try:
            sel = (model_choice or "Auto").strip().lower()
            if sel == 'ensemble':
                # Use learned ensemble fusion when we have at least one of LSTM/Transformer
                models = {}
                if 'lstm' in self.models:
                    models['lstm'] = self.models['lstm']
                if 'transformer' in self.models:
                    models['transformer'] = self.models['transformer']
                if models:
                    return _gen_ensemble(models, start, self.word_to_index, self.index_to_word, int(num_tokens), float(temperature), self.device)
                # Fallback to auto if ensemble unavailable
            mtype, model = self._resolve_model(model_choice)
            out = _gen_simple(model, start, self.word_to_index, self.index_to_word, int(num_tokens), float(temperature), self.device, model_type=mtype)
            return out
        finally:
            cfg.DECODING = old['DECODING']
            cfg.TOP_K = old['TOP_K']
            cfg.TOP_P = old['TOP_P']
            try:
                cfg.EARLY_STOP = old['EARLY_STOP']
                cfg.EOS_TOKEN = old['EOS_TOKEN']
                cfg.MIN_LENGTH = old['MIN_LENGTH']
            except Exception:
                pass

    # Streaming generation (yields chunks)
    def run_stream(self, prompt: str, num_tokens: int, temperature: float, decoding: str, top_k: int, top_p: float, model_choice: str = "Auto") -> Generator[str, None, None]:
        if not prompt:
            return
        toks = data.preprocess_data(prompt)
        start = " ".join(toks)
        old = {
            'DECODING': getattr(cfg, 'DECODING', 'sampling'),
            'TOP_K': getattr(cfg, 'TOP_K', 0),
            'TOP_P': getattr(cfg, 'TOP_P', 1.0),
            'EARLY_STOP': getattr(cfg, 'EARLY_STOP', True),
            'EOS_TOKEN': getattr(cfg, 'EOS_TOKEN', None),
            'MIN_LENGTH': getattr(cfg, 'MIN_LENGTH', 0),
        }
        cfg.DECODING = decoding
        cfg.TOP_K = int(top_k or 0)
        cfg.TOP_P = float(top_p or 1.0)
        if self.using_placeholder_vocab:
            try:
                cfg.EARLY_STOP = False
                cfg.EOS_TOKEN = None
                cfg.MIN_LENGTH = 0
            except Exception:
                pass
        try:
            sel = (model_choice or "Auto").strip().lower()
            # Streaming learned ensemble not implemented; prefer Transformer if available, else LSTM, else auto
            if sel == 'ensemble':
                prefer = 'transformer' if 'transformer' in self.models else ('lstm' if 'lstm' in self.models else self.model_type)
                mtype, model = self._resolve_model(prefer)
            else:
                mtype, model = self._resolve_model(model_choice)
            for tok in _gen_stream(model, start, self.word_to_index, self.index_to_word, int(num_tokens), float(temperature), self.device, model_type=mtype):
                yield tok
        finally:
            cfg.DECODING = old['DECODING']
            cfg.TOP_K = old['TOP_K']
            cfg.TOP_P = old['TOP_P']
            try:
                cfg.EARLY_STOP = old['EARLY_STOP']
                cfg.EOS_TOKEN = old['EOS_TOKEN']
                cfg.MIN_LENGTH = old['MIN_LENGTH']
            except Exception:
                pass

    def _resolve_model(self, model_choice: str):
        sel = (model_choice or "Auto").strip().lower()
        if sel == 'ensemble':
            # For now, prefer primary model when ensemble requested (ensemble text fusion lives elsewhere)
            sel = 'auto'
        # Explicit auto-best: recompute best from stats on demand
        if sel == 'auto_best' or ('auto' in sel and 'best' in sel):
            m_best = self._best_model_from_stats()
            if m_best and (m_best in self.models or self.ckpts.get(m_best)):
                if m_best not in self.models:
                    self._build_and_load(m_best, self.ckpts.get(m_best))
                return m_best, self.models[m_best]
            # Fallback to current auto
            sel = 'auto'
        if sel == 'auto':
            mtype = self.model_type
            model = self.models.get(mtype)
            return mtype, model
        else:
            mtype = sel
            if mtype not in self.models:
                self._build_and_load(mtype, self.ckpts.get(mtype))
            model = self.models[mtype]
            return mtype, model

# Singleton accessor
_runner: ActiveRunner | None = None

def get_runner() -> ActiveRunner:
    global _runner
    if _runner is None:
        _runner = ActiveRunner()
    return _runner

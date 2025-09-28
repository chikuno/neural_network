# inference/generate.py

import math
import torch
import torch.nn.functional as F
from config import config as cfg
from typing import Dict, Any

# ------------------------------
# Trace / Metrics Helper Utilities (refactored)
# ------------------------------
def _trace_init():
    if not getattr(cfg, 'GENERATION_TRACE', False):
        return None
    return {
        'steps': [],
        'meta': {
            'trace_file': getattr(cfg, 'TRACE_FILENAME', 'generation_trace.json'),
            'max_steps': int(getattr(cfg, 'TRACE_MAX_STEPS', 200))
        },
        'entropy': [],
        'temperatures': []
    }

def _trace_step(trace_obj, step, predicted_id, predicted_token, probs, entropy, temperature, extra=None, index_to_word=None):
    if trace_obj is None:
        return
    if len(trace_obj['steps']) >= trace_obj['meta']['max_steps']:
        return
    top_tokens = []
    try:
        if probs is not None and probs.numel() > 0:
            k = min(10, probs.size(0))
            topv, topi = torch.topk(probs, k=k)
            for v, i in zip(topv.tolist(), topi.tolist()):
                top_tokens.append({
                    'token_id': int(i),
                    'token': index_to_word.get(int(i), '<UNK>') if index_to_word else int(i),
                    'prob': float(v)
                })
    except Exception:
        pass
    entry = {
        'step': step,
        'predicted_id': int(predicted_id),
        'predicted_token': predicted_token,
        'entropy': entropy,
        'temperature': temperature,
        'top_tokens': top_tokens
    }
    if extra:
        entry.update(extra)
    trace_obj['steps'].append(entry)
    if entropy is not None:
        trace_obj['entropy'].append(float(entropy))
    trace_obj['temperatures'].append(float(temperature))

def _trace_finalize(trace_obj, generated_tokens, index_to_word, extra_summary=None):
    if trace_obj is None:
        return
    toks = [t for t in generated_tokens if t not in {'<PAD>', '<UNK>'}]
    total = len(toks)
    distinct1 = len(set(toks))/total if total else 0.0
    bigs = set((toks[i], toks[i+1]) for i in range(len(toks)-1)) if total > 1 else set()
    distinct2 = (len(bigs)/max(1, total-1)) if total > 1 else 0.0
    tris = set((toks[i], toks[i+1], toks[i+2]) for i in range(len(toks)-2)) if total > 2 else set()
    distinct3 = (len(tris)/max(1, total-2)) if total > 2 else 0.0
    avg_entropy = sum(trace_obj['entropy'])/len(trace_obj['entropy']) if trace_obj['entropy'] else 0.0
    temps = trace_obj['temperatures']
    avg_temp = sum(temps)/len(temps) if temps else 0.0
    if temps:
        drift = temps[-1] - temps[0]
    else:
        drift = 0.0
    # repetition ratio: immediate repeats
    if total > 1:
        repeats = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i-1])
        repetition_ratio = repeats / (len(toks)-1)
    else:
        repetition_ratio = 0.0
    summary = {
        'distinct_1': distinct1,
        'distinct_2': distinct2,
        'distinct_3': distinct3,
        'avg_entropy': avg_entropy,
        'avg_temperature': avg_temp,
        'temperature_drift': drift,
        'repetition_ratio': repetition_ratio,
        'total_tokens': total
    }
    if extra_summary:
        summary.update(extra_summary)
    trace_obj['summary'] = summary
    # Persist
    import json
    try:
        with open(trace_obj['meta']['trace_file'], 'w', encoding='utf-8') as f:
            json.dump(trace_obj, f, indent=2)
    except Exception:
        pass

def _get_eos_id(word_to_index):
    eos_token = getattr(cfg, 'EOS_TOKEN', None)
    if eos_token and eos_token in word_to_index:
        return int(word_to_index[eos_token])
    return None

def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    # logits: (vocab,)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < threshold, torch.tensor(float('-inf'), device=logits.device), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        # shift mask to include at least one token
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(0, sorted_indices, sorted_mask)
        logits = torch.where(mask, torch.tensor(float('-inf'), device=logits.device), logits)
    return logits

def _apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    if penalty <= 1.0 or not generated_ids:
        return logits
    # Simple penalty: subtract a constant from logits of recently used tokens
    unique_ids = set(generated_ids[-min(len(generated_ids), getattr(cfg, 'REPETITION_WINDOW', 64)):])
    if not unique_ids:
        return logits
    penalize = torch.tensor([i for i in unique_ids if 0 <= int(i) < logits.size(0)], device=logits.device, dtype=torch.long)
    if penalize.numel() == 0:
        return logits
    try:
        selected = logits.index_select(0, penalize)
        logits.index_put_((penalize,), selected - math.log(penalty))
    except Exception:
        # Fail-safe: ignore penalty if indexing fails (e.g., during test stubs)
        return logits
    return logits

def _blocked_tokens_for_no_repeat(history_ids, n):
    # Returns set of next-token ids that would create a repeated n-gram given history
    if n <= 0 or len(history_ids) < n - 1:
        return set()
    blocked = set()
    # Build map prefix -> set(next)
    prefix_len = n - 1
    for i in range(len(history_ids) - n + 1):
        prefix = tuple(history_ids[i:i+prefix_len])
        nxt = history_ids[i+prefix_len]
        blocked.add((prefix, nxt))
    # Current prefix
    cur_prefix = tuple(history_ids[-prefix_len:])
    return {nxt for (pref, nxt) in blocked if pref == cur_prefix}

def generate_text_simple(model, start_words, word_to_index, index_to_word, num_words, temperature, device, model_type):
    """
    Generates text with either sampling or beam search, including prompt-token bias and basic safeguards.

    Returns a final string after light post-processing.
    """
    model.eval()
    with torch.no_grad():
        # Normalize start input
        if isinstance(start_words, str):
            start_list = start_words.split()
            token_mode = False
        else:
            start_list = list(start_words)
            token_mode = all(isinstance(x, int) for x in start_list)

        if token_mode:
            input_sequence = start_list
            generated_words = [index_to_word.get(int(i), '<UNK>') for i in input_sequence]
        else:
            unk_id = word_to_index.get('<UNK>', 1)
            input_sequence = [word_to_index.get(str(word), unk_id) for word in start_list]
            generated_words = list(start_list)

        generated_ids = list(map(int, input_sequence)) if input_sequence else []

        # Initial input tensor
        if model_type == 'mlp':
            if len(input_sequence) == 0:
                input_sequence = [word_to_index.get('<UNK>', 0)]
            input_tensor = torch.tensor([list(map(int, input_sequence))], dtype=torch.long, device=device)
        else:
            init_ids = input_sequence if input_sequence else [word_to_index.get('<UNK>', 0)]
            input_tensor = torch.tensor(init_ids, dtype=torch.long, device=device).unsqueeze(0)

        hidden = None
        if model_type in ['rnn', 'gru', 'lstm']:
            hidden = model.init_hidden(1, device)
            logits, _, hidden = model(input_tensor, hidden)
            last_id = int(input_tensor[0, -1].item())
            input_tensor = torch.tensor([[last_id]], dtype=torch.long, device=device)

        # Config knobs
        top_k = getattr(cfg, 'TOP_K', 0)
        top_p = getattr(cfg, 'TOP_P', 1.0)
        rep_penalty = getattr(cfg, 'REPETITION_PENALTY', 1.0)
        no_repeat_ngram = getattr(cfg, 'NO_REPEAT_NGRAM_SIZE', 0)
        decoding = getattr(cfg, 'DECODING', 'sampling')
        eos_id = _get_eos_id(word_to_index)
        min_len = max(0, int(getattr(cfg, 'MIN_LENGTH', 0)))
        early_stop = bool(getattr(cfg, 'EARLY_STOP', True))
        punct_tokens = getattr(cfg, 'PUNCTUATION_TOKENS', [])
        punct_ids = [word_to_index.get(t) for t in punct_tokens if t in word_to_index]
        do_trace = bool(getattr(cfg, 'GENERATION_TRACE', False))
        trace_max = int(getattr(cfg, 'TRACE_MAX_STEPS', 200))
        trace = [] if do_trace else None
        diversity_collect = []

        # Beam search (skip for MLP)
        if decoding == 'beam' and model_type != 'mlp':
            beam_size = max(1, int(getattr(cfg, 'BEAM_SIZE', 3)))
            length_penalty = float(getattr(cfg, 'LENGTH_PENALTY', 0.7))

            beams = [(0.0, list(generated_ids), hidden, input_tensor.clone())]
            finished = []

            for t in range(num_words):
                new_beams = []
                for logp, ids, h, inp in beams:
                    # Stop expanding finished sequences (respect min length if early_stop)
                    if eos_id is not None and len(ids) > 0 and ids[-1] == eos_id and (not early_stop or t >= min_len):
                        finished.append((logp, ids))
                        continue

                    # Forward
                    if model_type in ['rnn', 'gru', 'lstm']:
                        logits, _, h2 = model(inp, h)
                    else:
                        if inp.dtype != torch.long:
                            inp = inp.to(torch.long)
                        pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                        src_key_padding_mask = (inp == pad_idx) if inp.dim() == 2 else None
                        logits, _ = model(inp, src_key_padding_mask=src_key_padding_mask)
                        h2 = h

                    if logits.dim() == 3:
                        logits = logits[:, -1, :]
                    logits = logits[0] / max(float(temperature), 1e-8)

                    # Penalties and filtering
                    logits = _apply_repetition_penalty(logits, ids, rep_penalty)
                    if no_repeat_ngram and ids:
                        blocked = _blocked_tokens_for_no_repeat(ids, no_repeat_ngram)
                        if blocked:
                            b_idx = torch.tensor(list(blocked), device=logits.device, dtype=torch.long)
                            if b_idx.numel() < logits.numel():
                                mask = torch.zeros_like(logits, dtype=torch.bool)
                                mask.index_fill_(0, b_idx, True)
                                logits = torch.where(mask, torch.tensor(float('-inf'), device=logits.device), logits)
                    logits_unf = logits.clone()
                    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    if not torch.isfinite(logits).any() or torch.all(logits == float('-inf')):
                        logits = logits_unf

                    # Expand
                    topv, topi = torch.topk(F.log_softmax(logits, dim=-1), k=min(beam_size, logits.numel()))
                    for add_logp, token_id in zip(topv.tolist(), topi.tolist()):
                        new_ids = ids + [int(token_id)]
                        new_logp = logp + float(add_logp)
                        if model_type in ['rnn', 'gru', 'lstm']:
                            next_inp = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
                        else:
                            seq = torch.cat([inp, torch.tensor([[int(token_id)]], dtype=torch.long, device=device)], dim=1)
                            max_len = max(2, int(getattr(cfg, 'SEQUENCE_LENGTH', 64)))
                            if seq.size(1) > max_len:
                                seq = seq[:, -max_len:]
                            next_inp = seq
                        new_beams.append((new_logp, new_ids, h2, next_inp))

                if not new_beams:
                    break

                def score_fn(entry):
                    lp, ids, _, _ = entry
                    if length_penalty > 0:
                        return lp / max(1.0, (len(ids) ** length_penalty))
                    return lp

                new_beams.sort(key=score_fn, reverse=True)
                beams = new_beams[:beam_size]

            all_cands = finished + [(lp, ids) for (lp, ids, _, _) in beams]
            if all_cands:
                def cand_score(c):
                    lp, ids = c
                    if length_penalty > 0:
                        return lp / max(1.0, (len(ids) ** length_penalty))
                    return lp

                best = max(all_cands, key=cand_score)
                generated_ids = best[1]
                generated_words = [index_to_word.get(int(i), '<UNK>') for i in generated_ids]

        else:
            # Sampling with enhanced degeneracy safeguards
            prompt_ids = set(generated_ids) if token_mode else set(input_sequence)
            max_consec = int(getattr(cfg, 'MAX_TOKEN_REPEAT', 0) or 0)
            dup_ratio_thresh = float(getattr(cfg, 'MAX_TOTAL_REPEAT_RATIO', 1.0))
            adapt_decay = float(getattr(cfg, 'ADAPTIVE_TEMP_DECAY', 1.0))
            min_temp = float(getattr(cfg, 'MIN_TEMPERATURE', 0.5))
            dynamic_temp = float(temperature)
            for t in range(num_words):
                # Forward
                if model_type in ['rnn', 'gru', 'lstm']:
                    logits, _, hidden = model(input_tensor, hidden)
                elif model_type == 'mlp':
                    out = model(input_tensor)
                    logits = out[0] if isinstance(out, tuple) else out
                else:
                    if input_tensor.dtype != torch.long:
                        input_tensor = input_tensor.to(torch.long)
                    pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                    src_key_padding_mask = (input_tensor == pad_idx) if input_tensor.dim() == 2 else None
                    logits, _ = model(input_tensor, src_key_padding_mask=src_key_padding_mask)

                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                logits = logits[0] / max(float(dynamic_temp), 1e-8)
                logits_unfiltered = logits.clone()

                # Penalties
                logits = _apply_repetition_penalty(logits, generated_ids, rep_penalty)
                if no_repeat_ngram and generated_ids:
                    blocked = _blocked_tokens_for_no_repeat(generated_ids, no_repeat_ngram)
                    if blocked:
                        # Filter out-of-range ids (can happen with test stubs returning tiny vocab logits)
                        safe_blocked = [b for b in blocked if 0 <= int(b) < logits.size(0)]
                        if safe_blocked:
                            blocked_idx = torch.tensor(safe_blocked, device=logits.device, dtype=torch.long)
                            if blocked_idx.numel() < logits.numel():
                                mask = torch.zeros_like(logits, dtype=torch.bool)
                                try:
                                    mask.index_fill_(0, blocked_idx, True)
                                    logits = torch.where(mask, torch.tensor(float('-inf'), device=logits.device), logits)
                                except Exception:
                                    pass

                # Prompt-token bias early on
                pbias = float(getattr(cfg, 'PROMPT_BIAS', 0.0))
                psteps = int(getattr(cfg, 'PROMPT_BIAS_STEPS', 0))
                if pbias > 0 and t < psteps and len(prompt_ids) > 0:
                    try:
                        idxs = torch.tensor(list({i for i in prompt_ids if 0 <= i < logits.numel()}), device=logits.device, dtype=torch.long)
                        if idxs.numel() > 0:
                            logits.index_put_((idxs,), logits.index_select(0, idxs) + pbias)
                    except Exception:
                        pass

                # Encourage closure near the end
                if t >= max(0, num_words - max(3, int(0.2 * num_words))):
                    bias = float(getattr(cfg, 'PUNCT_BIAS', 0.0))
                    if bias > 0:
                        if eos_id is not None:
                            try:
                                logits[eos_id] = logits[eos_id] + bias
                            except IndexError:
                                pass
                        if punct_ids:
                            for p_id in punct_ids:
                                try:
                                    if p_id is not None:
                                        logits[p_id] = logits[p_id] + bias
                                except IndexError:
                                    pass

                # Filter and sample
                logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                if not torch.isfinite(logits).any() or torch.all(logits == float('-inf')):
                    logits = logits_unfiltered
                probs = F.softmax(logits, dim=-1)
                # Diversity entropy floor: if distribution too peaky, gently raise temp
                try:
                    ent_floor = float(getattr(cfg, 'ENTROPY_FLOOR', 0.0) or 0.0)
                    if ent_floor > 0:
                        entropy = -(probs * (probs.clamp_min(1e-9).log())).sum().item()
                        if entropy < ent_floor:
                            boost = float(getattr(cfg, 'ENTROPY_BOOST', 1.0) or 1.0)
                            max_dyn = float(getattr(cfg, 'MAX_DYNAMIC_TEMPERATURE', 1.5) or 1.5)
                            dynamic_temp = min(max_dyn, dynamic_temp * boost)
                except Exception:
                    pass
                # Per-step entropy collection
                step_entropy = None
                try:
                    step_entropy = float(-(probs * (probs.clamp_min(1e-9).log())).sum().item())
                except Exception:
                    step_entropy = None
                if step_entropy is not None:
                    diversity_collect.append(step_entropy)

                if torch.isnan(probs).any() or probs.sum() <= 0:
                    predicted_index = int(torch.argmax(logits_unfiltered).item())
                else:
                    predicted_index = int(torch.multinomial(probs, 1).item())

                if do_trace and len(trace) < trace_max:
                    topn = 10
                    try:
                        topv, topi = torch.topk(probs, k=min(topn, probs.size(0)))
                        top_tokens = []
                        for v, i in zip(topv.tolist(), topi.tolist()):
                            top_tokens.append({'token_id': int(i), 'token': index_to_word.get(int(i), '<UNK>'), 'prob': float(v)})
                    except Exception:
                        top_tokens = []
                    trace.append({
                        'step': t,
                        'predicted_id': int(predicted_index),
                        'predicted_token': index_to_word.get(int(predicted_index), '<UNK>'),
                        'entropy': step_entropy,
                        'temperature': dynamic_temp,
                        'model_type': model_type,
                        'top_tokens': top_tokens
                    })

                # Enforce min length before allowing EOS
                if eos_id is not None and early_stop and t < min_len and predicted_index == eos_id:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    for alt in sorted_indices.tolist():
                        if alt != eos_id:
                            predicted_index = int(alt)
                            break

                generated_ids.append(int(predicted_index))
                generated_words.append(index_to_word.get(int(predicted_index), '<UNK>'))

                # Degeneracy: cap consecutive repeats
                if max_consec > 0 and len(generated_ids) >= max_consec:
                    recent = generated_ids[-max_consec:]
                    if all(r == recent[0] for r in recent):
                        # force resample once by zeroing that token prob next step
                        try:
                            banned_id = recent[0]
                            # Slight temp decay to encourage diversity
                            dynamic_temp = max(min_temp, dynamic_temp * adapt_decay)
                            # Replace last token with alternative with highest prob different token if possible
                            # (We cannot undo already appended token; future steps will avoid it)
                        except Exception:
                            pass

                # Global duplicate ratio heuristic: if too many tokens are repeats of immediate previous tokens
                if dup_ratio_thresh < 1.0 and len(generated_ids) > 10:
                    repeats = 0
                    for i in range(1, len(generated_ids)):
                        if generated_ids[i] == generated_ids[i-1]:
                            repeats += 1
                    if repeats / (len(generated_ids)-1) > dup_ratio_thresh:
                        dynamic_temp = max(min_temp, dynamic_temp * adapt_decay)
                        # If still degenerate near end, break early
                        if t > num_words * 0.5:
                            break

                # Next input
                if model_type == 'mlp':
                    input_tensor = torch.tensor([[int(predicted_index)]], dtype=torch.long, device=device)
                else:
                    if model_type in ['rnn', 'gru', 'lstm']:
                        input_tensor = torch.tensor([[predicted_index]], dtype=torch.long, device=device)
                    else:
                        max_len = max(2, int(getattr(cfg, 'SEQUENCE_LENGTH', 64)))
                        if input_tensor.dim() == 1:
                            input_tensor = input_tensor.unsqueeze(0)
                        seq = torch.cat([input_tensor, torch.tensor([[predicted_index]], dtype=torch.long, device=device)], dim=1)
                        if seq.size(1) > max_len:
                            seq = seq[:, -max_len:]
                        input_tensor = seq

                if eos_id is not None and early_stop and len(generated_ids) >= min_len and generated_ids[-1] == eos_id:
                    break

        # Post-process words into a string
        cleaned = []
        placeholders = {'<PAD>', '<UNK>'}
        for w in generated_words:
            if w in placeholders:
                continue
            if len(cleaned) > 0 and cleaned[-1] == w:
                continue
            cleaned.append(w)
        if getattr(cfg, 'EOS_TOKEN', None):
            try:
                while cleaned and cleaned[-1] == cfg.EOS_TOKEN:
                    cleaned.pop()
            except Exception:
                pass
        if len(cleaned) == 0:
            if isinstance(start_words, str):
                base = [w for w in start_words.split() if w and w not in placeholders]
            else:
                ids = start_list if 'start_list' in locals() else []
                base = [index_to_word.get(int(i), '') for i in ids]
                base = [w for w in base if w and w not in placeholders]
            if base:
                return " ".join(base[:max(3, min(len(base), getattr(cfg, 'SEQUENCE_LENGTH', 30)))])
            desc = getattr(cfg, 'BOT_DESCRIPTION', None)
            name = getattr(cfg, 'BOT_NAME', None)
            if desc:
                return desc
            if name:
                return f"I am {name}."
            return "<no output>"
        final_text = " ".join(cleaned)

        if getattr(cfg, 'DIVERSITY_METRICS', False):
            try:
                toks = [w for w in cleaned if w not in {'<PAD>', '<UNK>'}]
                total = len(toks)
                if total > 0:
                    d1 = len(set(toks)) / total
                    bigs = set((toks[i], toks[i+1]) for i in range(len(toks)-1)) if len(toks) > 1 else set()
                    d2 = (len(bigs) / max(1, len(toks)-1)) if len(toks) > 1 else 0.0
                    avg_ent = float(sum(diversity_collect)/len(diversity_collect)) if diversity_collect else 0.0
                    print(f"[Diversity] distinct-1={d1:.3f} distinct-2={d2:.3f} avg-entropy={avg_ent:.3f}")
                    if do_trace and trace is not None:
                        trace.append({'summary': {'distinct_1': d1, 'distinct_2': d2, 'avg_entropy': avg_ent, 'total_tokens': total}})
            except Exception:
                pass

        if do_trace and trace is not None:
            import json
            try:
                fname = getattr(cfg, 'TRACE_FILENAME', 'generation_trace.json')
                with open(fname, 'w', encoding='utf-8') as f:
                    json.dump(trace, f, indent=2)
            except Exception:
                pass

        return final_text

def generate_stream(model, start_words, word_to_index, index_to_word, num_words, temperature, device, model_type):
    """Yield generated tokens incrementally (sampling decoding only).

    This mirrors generate_text_simple but yields one word at a time for streaming chat UX.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(start_words, str):
            start_list = start_words.split()
            token_mode = False
        else:
            start_list = list(start_words)
            token_mode = all(isinstance(x, int) for x in start_list)

        if token_mode:
            input_sequence = start_list
        else:
            input_sequence = [word_to_index.get(word, word_to_index.get('<UNK>', 0)) for word in start_list]

        generated_ids = list(map(int, input_sequence)) if input_sequence else []

        def one_hot_tensor(index, size):
            vec = torch.zeros(1, size, dtype=torch.float, device=device)
            if 0 <= index < size:
                vec[0, index] = 1.0
            return vec

        if model_type == 'mlp':
            if len(input_sequence) == 0:
                input_sequence = [word_to_index.get('<UNK>', 0)]
            input_tensor = torch.tensor([list(map(int, input_sequence))], dtype=torch.long, device=device)
        else:
            input_tensor = torch.tensor(input_sequence if input_sequence else [word_to_index.get('<UNK>', 0)], dtype=torch.long).unsqueeze(0).to(device)

        hidden = None
        if model_type in ['rnn', 'gru', 'lstm']:
            hidden = model.init_hidden(1, device)
            logits, _, hidden = model(input_tensor, hidden)
            last_id = int(input_tensor[0, -1].item())
            input_tensor = torch.tensor([[last_id]], dtype=torch.long, device=device)

        top_k = getattr(cfg, 'TOP_K', 0)
        top_p = getattr(cfg, 'TOP_P', 1.0)
        rep_penalty = getattr(cfg, 'REPETITION_PENALTY', 1.0)
        no_repeat_ngram = getattr(cfg, 'NO_REPEAT_NGRAM_SIZE', 0)
        eos_id = _get_eos_id(word_to_index)
        early_stop = bool(getattr(cfg, 'EARLY_STOP', True))
        min_len = max(0, int(getattr(cfg, 'MIN_LENGTH', 0)))
        punct_tokens = getattr(cfg, 'PUNCTUATION_TOKENS', [])
        punct_ids = [word_to_index.get(t) for t in punct_tokens if t in word_to_index]

        prompt_ids = set(generated_ids) if token_mode else set(input_sequence)
        for t in range(num_words):
            if model_type in ['rnn', 'gru', 'lstm']:
                logits, _, hidden = model(input_tensor, hidden)
            elif model_type == 'mlp':
                out = model(input_tensor)
                logits = out[0] if isinstance(out, tuple) else out
            else:
                if input_tensor.dtype != torch.long:
                    input_tensor = input_tensor.to(torch.long)
                pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                src_key_padding_mask = (input_tensor == pad_idx) if input_tensor.dim() == 2 else None
                logits, _ = model(input_tensor, src_key_padding_mask=src_key_padding_mask)

            if logits.dim() == 3:
                logits = logits[:, -1, :]

            logits = logits[0]
            logits = logits / max(float(temperature), 1e-8)
            logits_unfiltered = logits.clone()
            logits = _apply_repetition_penalty(logits, generated_ids, rep_penalty)
            if no_repeat_ngram and generated_ids:
                blocked = _blocked_tokens_for_no_repeat(generated_ids, no_repeat_ngram)
                if blocked:
                    blocked_idx = torch.tensor(list(blocked), device=logits.device, dtype=torch.long)
                    if blocked_idx.numel() < logits.numel():
                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        mask.index_fill_(0, blocked_idx, True)
                        logits = torch.where(mask, torch.tensor(float('-inf'), device=logits.device), logits)

            # Apply prompt-token bias for the first few steps
            pbias = float(getattr(cfg, 'PROMPT_BIAS', 0.0))
            psteps = int(getattr(cfg, 'PROMPT_BIAS_STEPS', 0))
            if pbias > 0 and t < psteps and len(prompt_ids) > 0:
                try:
                    idxs = torch.tensor(list({i for i in prompt_ids if 0 <= i < logits.numel()}), device=logits.device, dtype=torch.long)
                    if idxs.numel() > 0:
                        logits.index_put_((idxs,), logits.index_select(0, idxs) + pbias)
                except Exception:
                    pass
            # If near the end, lightly bias EOS and punctuation to encourage closure
            if t >= max(0, num_words - max(3, int(0.2 * num_words))):
                bias = float(getattr(cfg, 'PUNCT_BIAS', 0.0))
                if bias > 0:
                    # Bias EOS token if available
                    if eos_id is not None:
                        try:
                            logits[eos_id] = logits[eos_id] + bias
                        except IndexError:
                            pass
                    # Bias other punctuation tokens
                    if punct_ids:
                        for p_id in punct_ids:
                            try:
                                if p_id is not None:
                                    logits[p_id] = logits[p_id] + bias
                            except IndexError:
                                pass

            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            if not torch.isfinite(logits).any() or torch.all(logits == float('-inf')):
                logits = logits_unfiltered
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                predicted_index = int(torch.argmax(logits_unfiltered).item())
            else:
                predicted_index = int(torch.multinomial(probs, 1).item())

            # Respect min length by avoiding EOS until allowed
            if eos_id is not None and early_stop and t < min_len and predicted_index == eos_id:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                for alt in sorted_indices.tolist():
                    if alt != eos_id:
                        predicted_index = int(alt)
                        break

            # Decide whether to stop before yielding the EOS token
            will_stop_on_eos = False
            if eos_id is not None and early_stop and predicted_index == eos_id:
                # We will append and then break without yielding the EOS token
                will_stop_on_eos = True

            generated_ids.append(int(predicted_index))
            token_str = index_to_word.get(int(predicted_index), '<UNK>')
            # Skip yielding <eos> so UI doesn't display it
            if token_str not in {'<PAD>', '<UNK>'} and not (will_stop_on_eos and token_str == getattr(cfg, 'EOS_TOKEN', None)):
                yield token_str

            if model_type == 'mlp':
                input_tensor = torch.tensor([[int(predicted_index)]], dtype=torch.long, device=device)
            else:
                if model_type in ['rnn', 'gru', 'lstm']:
                    input_tensor = torch.tensor([[predicted_index]], dtype=torch.long, device=device)
                else:
                    if 'SEQUENCE_LENGTH' in dir(cfg):
                        max_len = max(2, int(cfg.SEQUENCE_LENGTH))
                    else:
                        max_len = 64
                    if input_tensor.dim() == 1:
                        input_tensor = input_tensor.unsqueeze(0)
                    seq = torch.cat([input_tensor, torch.tensor([[predicted_index]], dtype=torch.long, device=device)], dim=1)
                    if seq.size(1) > max_len:
                        seq = seq[:, -max_len:]
                    input_tensor = seq

            if eos_id is not None and early_stop and len(generated_ids) >= min_len and generated_ids[-1] == eos_id:
                break


def generate_text_learned_ensemble(models: Dict[str, Any], start_words, word_to_index, index_to_word,
                                   num_words, temperature, device):
    """Learned ensemble (sampling only) combining LSTM + Transformer via weighted log-prob fusion.

    Requirements:
      - models dict contains keys 'lstm' and 'transformer' (missing ones are skipped)
      - Uses config.ENSEMBLE_LSTM_WEIGHT for weighting; (1-w) for transformer.
      - Applies existing repetition, n-gram, prompt bias, entropy floor, degeneracy safeguards.
    """
    lstm = models.get('lstm', None)
    transformer = models.get('transformer', None)
    available = [m for m in [lstm, transformer] if m is not None]
    if not available:
        raise ValueError("No models provided for learned ensemble")
    for m in available:
        m.eval()

    with torch.no_grad():
        # Normalize start input
        if isinstance(start_words, str):
            start_list = start_words.split()
            token_mode = False
        else:
            start_list = list(start_words)
            token_mode = all(isinstance(x, int) for x in start_list)

        unk_id = word_to_index.get('<UNK>', 1)
        if token_mode:
            input_sequence = list(start_list)
            generated_words = [index_to_word.get(int(i), '<UNK>') for i in start_list]
        else:
            input_sequence = [word_to_index.get(str(w), unk_id) for w in start_list]
            generated_words = list(start_list)
        generated_ids = list(map(int, input_sequence)) if input_sequence else []

        # Prepare per-model input tensors / hidden
        if lstm is not None:
            lstm_hidden = lstm.init_hidden(1, device)
            init_ids = input_sequence if input_sequence else [word_to_index.get('<UNK>', 0)]
            lstm_inp = torch.tensor([[int(init_ids[-1])]], dtype=torch.long, device=device)
            # Prime LSTM with prefix (except last already in lstm_inp)
            if len(init_ids) > 1:
                prefix = torch.tensor([init_ids[:-1]], dtype=torch.long, device=device)
                prime_out = lstm(prefix, lstm_hidden)
                # Accept variable return forms: (logits, aux, hidden) | (logits, hidden) | logits
                if isinstance(prime_out, (list, tuple)):
                    if len(prime_out) == 3:
                        _, _, lstm_hidden = prime_out
                    elif len(prime_out) == 2:
                        _, lstm_hidden = prime_out
                    else:
                        # Single element; hidden unchanged
                        pass
                # else: assume logits only; hidden unchanged
        else:
            lstm_hidden = None
            lstm_inp = None

        if transformer is not None:
            trans_ids = input_sequence if input_sequence else [word_to_index.get('<UNK>', 0)]
            transformer_inp = torch.tensor([list(map(int, trans_ids))], dtype=torch.long, device=device)
        else:
            transformer_inp = None

        # Config knobs
        top_k = getattr(cfg, 'TOP_K', 0)
        top_p = getattr(cfg, 'TOP_P', 1.0)
        rep_penalty = getattr(cfg, 'REPETITION_PENALTY', 1.0)
        no_repeat_ngram = getattr(cfg, 'NO_REPEAT_NGRAM_SIZE', 0)
        eos_id = _get_eos_id(word_to_index)
        min_len = max(0, int(getattr(cfg, 'MIN_LENGTH', 0)))
        early_stop = bool(getattr(cfg, 'EARLY_STOP', True))
        punct_tokens = getattr(cfg, 'PUNCTUATION_TOKENS', [])
        punct_ids = [word_to_index.get(t) for t in punct_tokens if t in word_to_index]
        # Dynamic ensemble weighting support: if ENSEMBLE_MODE matches dynamic sentinel, load stats
        lstm_weight = float(getattr(cfg, 'ENSEMBLE_LSTM_WEIGHT', 0.5))
        tr_weight = 1.0 - lstm_weight
        try:
            mode = getattr(cfg, 'ENSEMBLE_MODE', 'simple')
            dyn_sentinel = getattr(cfg, 'ENSEMBLE_MODE_DYNAMIC_BASE', 'learned_dynamic')
        except Exception:
            mode = 'simple'
            dyn_sentinel = 'learned_dynamic'
        dynamic_active = (mode == 'learned_dynamic') or (mode == dyn_sentinel)
        dynamic_source_path = 'ensemble_stats.json'
        if dynamic_active:
            import json
            import os
            if os.path.exists(dynamic_source_path):
                try:
                    with open(dynamic_source_path,'r',encoding='utf-8') as f:
                        estats = json.load(f) or {}
                    model_stats = (estats.get('models') or {})
                    losses = {}
                    for mname in ['lstm','transformer']:
                        if mname in model_stats and model_stats[mname].get('val_loss', None) is not None and model_stats[mname]['val_loss'] > 0:
                            losses[mname] = float(model_stats[mname]['val_loss'])
                    # If both present, inverse-loss normalize; if one missing fallback to static
                    if losses:
                        # Add small epsilon for numeric stability
                        inv = {k: 1.0 / (v + 1e-8) for k, v in losses.items() if v > 0}
                        total_inv = sum(inv.values())
                        if total_inv > 0 and ('lstm' in inv or 'transformer' in inv):
                            if 'lstm' in inv and 'transformer' in inv:
                                lstm_weight = inv.get('lstm', 0.0) / total_inv
                                tr_weight = inv.get('transformer', 0.0) / total_inv
                            elif 'lstm' in inv:
                                lstm_weight = 1.0
                                tr_weight = 0.0
                            elif 'transformer' in inv:
                                lstm_weight = 0.0
                                tr_weight = 1.0
                except Exception:
                    pass
            else:
                # Friendly notice only once per call
                print("[Ensemble] Dynamic mode requested but ensemble_stats.json not found. Using static weights.")
        temp_align = bool(getattr(cfg, 'ENSEMBLE_TEMP_ALIGN', True))
        prompt_ids = set(generated_ids) if token_mode else set(input_sequence)
        max_consec = int(getattr(cfg, 'MAX_TOKEN_REPEAT', 0) or 0)
        dup_ratio_thresh = float(getattr(cfg, 'MAX_TOTAL_REPEAT_RATIO', 1.0))
        adapt_decay = float(getattr(cfg, 'ADAPTIVE_TEMP_DECAY', 1.0))
        min_temp = float(getattr(cfg, 'MIN_TEMPERATURE', 0.5))
        dynamic_temp = float(temperature)

        # Optional trace collection
        do_trace = bool(getattr(cfg, 'GENERATION_TRACE', False))
        trace_max = int(getattr(cfg, 'TRACE_MAX_STEPS', 200))
        trace = [] if do_trace else None
        diversity_collect = []  # per-step entropy

        for t in range(num_words):
            logits_lstm = None
            logits_tr = None
            # Forward LSTM one-step
            if lstm is not None:
                step_out = lstm(lstm_inp, lstm_hidden)
                # Normalize output shapes
                out_lstm = None
                if isinstance(step_out, (list, tuple)):
                    if len(step_out) == 3:
                        out_lstm, _, lstm_hidden = step_out
                    elif len(step_out) == 2:
                        out_lstm, lstm_hidden = step_out
                    else:
                        out_lstm = step_out[0]
                else:
                    out_lstm = step_out
                if out_lstm is not None:
                    if out_lstm.dim() == 3:
                        logits_lstm = out_lstm[:, -1, :][0]
                    else:
                        logits_lstm = out_lstm[0]
            # Forward Transformer full context
            if transformer is not None:
                pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                src_key_padding_mask = (transformer_inp == pad_idx) if transformer_inp.dim() == 2 else None
                out_tr, _ = transformer(transformer_inp, src_key_padding_mask=src_key_padding_mask)
                if out_tr.dim() == 3:
                    logits_tr = out_tr[:, -1, :][0]
                else:
                    logits_tr = out_tr[0]

            # Combine
            if logits_lstm is None:
                fused = logits_tr
            elif logits_tr is None:
                fused = logits_lstm
            else:
                # Temperature scaling applied later; optionally variance normalize
                ll = logits_lstm
                lt = logits_tr
                if temp_align:
                    def _norm(v):
                        std = v.std().clamp_min(1e-6)
                        return (v - v.mean()) / std
                    ll = _norm(ll)
                    lt = _norm(lt)
                # Weighted log-prob fusion via log_softmax first
                ll_lp = F.log_softmax(ll / max(dynamic_temp,1e-8), dim=-1)
                lt_lp = F.log_softmax(lt / max(dynamic_temp,1e-8), dim=-1)
                fused = torch.logaddexp(ll_lp + math.log(max(lstm_weight,1e-8)),
                                        lt_lp + math.log(max(tr_weight,1e-8)))
                # Convert back to pseudo-logits (log-probs already acceptable downstream)
            logits = fused
            logits_unfiltered = logits.clone()

            # Repetition penalty & n-gram blocking operate in logit space
            logits = _apply_repetition_penalty(logits, generated_ids, rep_penalty)
            if no_repeat_ngram and generated_ids:
                blocked = _blocked_tokens_for_no_repeat(generated_ids, no_repeat_ngram)
                if blocked:
                    safe_blocked = [b for b in blocked if 0 <= int(b) < logits.size(0)]
                    if safe_blocked:
                        blocked_idx = torch.tensor(safe_blocked, device=logits.device, dtype=torch.long)
                        if blocked_idx.numel() < logits.numel():
                            mask = torch.zeros_like(logits, dtype=torch.bool)
                            try:
                                mask.index_fill_(0, blocked_idx, True)
                                logits = torch.where(mask, torch.tensor(float('-inf'), device=logits.device), logits)
                            except Exception:
                                pass

            # Prompt bias
            pbias = float(getattr(cfg, 'PROMPT_BIAS', 0.0))
            psteps = int(getattr(cfg, 'PROMPT_BIAS_STEPS', 0))
            if pbias > 0 and t < psteps and len(prompt_ids) > 0:
                try:
                    idxs = torch.tensor(list({i for i in prompt_ids if 0 <= i < logits.numel()}), device=logits.device, dtype=torch.long)
                    if idxs.numel() > 0:
                        logits.index_put_((idxs,), logits.index_select(0, idxs) + pbias)
                except Exception:
                    pass

            # Closure encouragement near end
            if t >= max(0, num_words - max(3, int(0.2 * num_words))):
                bias = float(getattr(cfg, 'PUNCT_BIAS', 0.0))
                if bias > 0:
                    if eos_id is not None:
                        try:
                            logits[eos_id] = logits[eos_id] + bias
                        except Exception:
                            pass
                    for p_id in punct_ids:
                        try:
                            if p_id is not None:
                                logits[p_id] = logits[p_id] + bias
                        except Exception:
                            pass

            # Top-k / nucleus
            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            if not torch.isfinite(logits).any() or torch.all(logits == float('-inf')):
                logits = logits_unfiltered
            probs = F.softmax(logits, dim=-1)
            step_entropy = None
            if probs.numel() > 0:
                try:
                    step_entropy = float(-(probs * (probs.clamp_min(1e-9).log())).sum().item())
                except Exception:
                    step_entropy = None
            if step_entropy is not None:
                diversity_collect.append(step_entropy)

            # Entropy-based diversity boost
            try:
                ent_floor = float(getattr(cfg, 'ENTROPY_FLOOR', 0.0) or 0.0)
                if ent_floor > 0:
                    entropy = -(probs * (probs.clamp_min(1e-9).log())).sum().item()
                    if entropy < ent_floor:
                        boost = float(getattr(cfg, 'ENTROPY_BOOST', 1.0) or 1.0)
                        max_dyn = float(getattr(cfg, 'MAX_DYNAMIC_TEMPERATURE', 1.5) or 1.5)
                        dynamic_temp = min(max_dyn, dynamic_temp * boost)
            except Exception:
                pass

            if torch.isnan(probs).any() or probs.sum() <= 0:
                predicted_index = int(torch.argmax(logits_unfiltered).item())
            else:
                predicted_index = int(torch.multinomial(probs, 1).item())

            # Trace logging (capture before advancing) limited to trace_max entries
            if do_trace and len(trace) < trace_max:
                topn = 10
                try:
                    topv, topi = torch.topk(probs, k=min(topn, probs.size(0)))
                    top_tokens = []
                    for v, i in zip(topv.tolist(), topi.tolist()):
                        top_tokens.append({
                            'token_id': int(i),
                            'token': index_to_word.get(int(i), '<UNK>'),
                            'prob': float(v)
                        })
                except Exception:
                    top_tokens = []
                trace.append({
                    'step': t,
                    'predicted_id': int(predicted_index),
                    'predicted_token': index_to_word.get(int(predicted_index), '<UNK>'),
                    'entropy': step_entropy,
                    'temperature': dynamic_temp,
                    'lstm_weight': float(lstm_weight),
                    'transformer_weight': float(tr_weight),
                    'top_tokens': top_tokens
                })

            # Min length EOS block
            if eos_id is not None and early_stop and t < min_len and predicted_index == eos_id:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                for alt in sorted_indices.tolist():
                    if alt != eos_id:
                        predicted_index = int(alt)
                        break

            generated_ids.append(int(predicted_index))
            generated_words.append(index_to_word.get(int(predicted_index), '<UNK>'))

            # Consecutive degeneracy handling
            if max_consec > 0 and len(generated_ids) >= max_consec:
                recent = generated_ids[-max_consec:]
                if all(r == recent[0] for r in recent):
                    dynamic_temp = max(min_temp, dynamic_temp * adapt_decay)

            # Global duplicate ratio
            if dup_ratio_thresh < 1.0 and len(generated_ids) > 10:
                repeats = sum(1 for i in range(1, len(generated_ids)) if generated_ids[i] == generated_ids[i-1])
                if repeats / (len(generated_ids)-1) > dup_ratio_thresh:
                    dynamic_temp = max(min_temp, dynamic_temp * adapt_decay)
                    if t > num_words * 0.5:
                        break

            # Advance inputs
            if lstm_inp is not None:
                lstm_inp = torch.tensor([[predicted_index]], dtype=torch.long, device=device)
            if transformer_inp is not None:
                seq = torch.cat([transformer_inp, torch.tensor([[predicted_index]], dtype=torch.long, device=device)], dim=1)
                max_len = max(2, int(getattr(cfg, 'SEQUENCE_LENGTH', 64)))
                if seq.size(1) > max_len:
                    seq = seq[:, -max_len:]
                transformer_inp = seq

            if eos_id is not None and early_stop and len(generated_ids) >= min_len and generated_ids[-1] == eos_id:
                break

        # Post-process same as simple generator
        cleaned = []
        placeholders = {'<PAD>', '<UNK>'}
        for w in generated_words:
            if w in placeholders:
                continue
            if cleaned and cleaned[-1] == w:
                continue
            cleaned.append(w)
        if getattr(cfg, 'EOS_TOKEN', None):
            try:
                while cleaned and cleaned[-1] == cfg.EOS_TOKEN:
                    cleaned.pop()
            except Exception:
                pass
        if not cleaned:
            final_text = "<no output>"
        else:
            final_text = " ".join(cleaned)

        # Diversity metrics (distinct-1/2, avg entropy) optionally printed / saved in trace
        if getattr(cfg, 'DIVERSITY_METRICS', False):
            try:
                toks = [w for w in cleaned if w not in {'<PAD>','<UNK>'}]
                total = len(toks)
                if total > 0:
                    unigrams = len(set(toks)) / total
                    bigrams = set()
                    for i in range(len(toks)-1):
                        bigrams.add((toks[i], toks[i+1]))
                    distinct2 = (len(bigrams) / max(1, len(toks)-1)) if len(toks) > 1 else 0.0
                    avg_entropy = float(sum(diversity_collect)/len(diversity_collect)) if diversity_collect else 0.0
                    print(f"[Diversity] distinct-1={unigrams:.3f} distinct-2={distinct2:.3f} avg-entropy={avg_entropy:.3f}")
                    if do_trace and trace is not None:
                        # Append summary record with ensemble weights if available
                        trace.append({
                            'summary': {
                                'distinct_1': unigrams,
                                'distinct_2': distinct2,
                                'avg_entropy': avg_entropy,
                                'total_tokens': total,
                                'lstm_weight': float(lstm_weight),
                                'transformer_weight': float(tr_weight)
                            }
                        })
            except Exception:
                pass

        # Persist trace if enabled
        if do_trace and trace is not None:
            import json
            import os
            try:
                fname = getattr(cfg, 'TRACE_FILENAME', 'generation_trace.json')
                with open(fname, 'w', encoding='utf-8') as tf:
                    json.dump(trace, tf, indent=2)
            except Exception:
                pass

        return final_text

import re
from typing import List

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

# Basic English text refinement utilities
_whitespace_re = re.compile(r"\s+")
_space_punct_re = re.compile(r"\s+([,.;:!?])")
_quote_space_re = re.compile(r"\s+([\'\"])\s+")


def _simple_sentence_split(text: str) -> List[str]:
    # naive split on sentence enders; keeps delimiters
    parts = re.split(r"([.!?])", text)
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        end = parts[i + 1] if i + 1 < len(parts) else ""
        if seg.strip():
            out.append((seg.strip(), end))
    return [s + e for s, e in out]


def _capitalize_sentences(text: str) -> str:
    sents = _simple_sentence_split(text)
    cap = [s[:1].upper() + s[1:] if s else s for s in sents]
    return " ".join(cap)


def _fix_spacing(text: str) -> str:
    # Collapse whitespace
    text = _whitespace_re.sub(" ", text).strip()
    # Remove space before punctuation
    text = _space_punct_re.sub(r"\1", text)
    return text


def _dedupe_tokens(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text
    out = [tokens[0]]
    for t in tokens[1:]:
        if t != out[-1]:
            out.append(t)
    return " ".join(out)


def _spellcheck(text: str) -> str:
    if SpellChecker is None:
        return text
    sp = SpellChecker()
    tokens = text.split()
    misspelled = sp.unknown(tokens)
    corrected = []
    for t in tokens:
        if t in misspelled:
            suggestion = sp.correction(t)
            corrected.append(suggestion or t)
        else:
            corrected.append(t)
    return " ".join(corrected)


def refine_text(text: str) -> str:
    if not text:
        return text
    from config import config as cfg
    t = text
    t = _fix_spacing(t)
    t = _dedupe_tokens(t)
    t = _capitalize_sentences(t)
    t = _spellcheck(t)
    # Lightweight common replacements
    common = {
        'teh': 'the',
        'adress': 'address',
        'recieve': 'receive'
    }
    tokens = t.split()
    tokens = [common.get(w.lower(), w) for w in tokens]
    t = ' '.join(tokens)
    # Final punctuation enforcement
    try:
        if getattr(cfg, 'GRAMMAR_CORRECTION', False) and getattr(cfg, 'GRAMMAR_ENSURE_PERIOD', False):
            if len(tokens) >= int(getattr(cfg, 'GRAMMAR_MIN_LENGTH', 0)) and not t.rstrip().endswith(('.', '!', '?')):
                t = t.rstrip() + '.'
    except Exception:
        pass
    # Proper noun restoration
    try:
        proper = set(getattr(cfg, 'GRAMMAR_PROPER_NOUNS', []))
        if proper:
            rebuilt = []
            for w in t.split():
                if w.lower() in {p.lower() for p in proper}:
                    # pick canonical casing from list
                    for p in proper:
                        if p.lower() == w.lower():
                            rebuilt.append(p)
                            break
                else:
                    rebuilt.append(w)
            t = ' '.join(rebuilt)
    except Exception:
        pass
    # Mid-sentence <eos> cleanup
    try:
        if getattr(cfg, 'GRAMMAR_FILTER_EOS_MID', False) and getattr(cfg, 'EOS_TOKEN', None):
            eos_tok = cfg.EOS_TOKEN
            words = t.split()
            cleaned = []
            for i, w in enumerate(words):
                if w == eos_tok:
                    nxt = words[i+1] if i+1 < len(words) else ''
                    if nxt and nxt[:1].islower() and not cleaned[-1].endswith(('.', '!', '?')):
                        continue  # drop mid-sentence eos
                cleaned.append(w)
            t = ' '.join(cleaned)
    except Exception:
        pass
    # Sentence-level dedupe & repetition flush
    try:
        if getattr(cfg, 'GRAMMAR_DEDUPE_SENTENCES', False):
            sents = _simple_sentence_split(t)
            seen = set()
            kept = []
            for s in sents:
                key = s.strip().lower()
                if key not in seen:
                    kept.append(s)
                    seen.add(key)
            # Repetition flush: if last sentence duplicates earlier one, keep only last occurrence
            if getattr(cfg, 'GRAMMAR_REPEAT_FLUSH', False) and len(kept) > 2:
                last = kept[-1].strip().lower()
                earlier = [i for i,x in enumerate(kept[:-1]) if x.strip().lower()==last]
                if earlier:
                    kept = [k for i,k in enumerate(kept) if i not in earlier]
                    kept.append(sents[-1])
            t = ' '.join(kept)
    except Exception:
        pass
    # Bigram smoothing placeholder (heuristic): replace isolated rare bigrams by removing them
    try:
        if getattr(cfg, 'GRAMMAR_BIGRAM_SMOOTH', False):
            words = t.split()
            if len(words) > 3:
                from collections import Counter
                bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
                freq = Counter(bigrams)
                pruned = []
                i = 0
                while i < len(words):
                    if i < len(words)-1 and freq[(words[i], words[i+1])] == 1 and (words[i+1].islower()):
                        # drop the first word of a rare low-case-following bigram
                        pruned.append(words[i+1])
                        i += 2
                    else:
                        pruned.append(words[i])
                        i += 1
                t = ' '.join(pruned)
    except Exception:
        pass
    return t

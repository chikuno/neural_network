import os
import json
import math
import re
from collections import Counter

# Simple TF-IDF retrieval over sentence-level JSONL

class TfidfRetriever:
    def __init__(self, jsonl_path: str, min_chars: int = 20):
        self.docs = []
        self.df = Counter()
        self.N = 0
        self.min_chars = min_chars
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        text = (rec.get('text') or '').strip()
                        if len(text) >= self.min_chars:
                            self.docs.append(text)
                    except Exception:
                        continue
        self.N = len(self.docs)
        # Build document frequencies
        for s in self.docs:
            toks = set(self._tokenize(s))
            for t in toks:
                self.df[t] += 1

    def _tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if w]

    def _tf(self, tokens):
        cnt = Counter(tokens)
        L = max(1, len(tokens))
        return {k: v / L for k, v in cnt.items()}

    def _idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N + 1) / (df + 1)) + 1.0

    def _vec(self, text: str):
        toks = self._tokenize(text)
        tf = self._tf(toks)
        return {t: tf[t] * self._idf(t) for t in tf.keys()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        # dot
        dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in set(a.keys()) | set(b.keys()))
        # norms
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def top_k(self, query: str, k: int = 3):
        qv = self._vec(query)
        scored = []
        for s in self.docs:
            sv = self._vec(s)
            scored.append((self._cosine(qv, sv), s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

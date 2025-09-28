import os
import re
import logging
import torch
import requests
from collections import Counter
# ThreadPoolExecutor is a stdlib import; keep imports grouped at the top
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
# BeautifulSoup is required; wikipediaapi is optional and imported lazily
from bs4 import BeautifulSoup
# googletrans import is intentionally lazy inside translate_text()
# because importing it at module import time pulls in httpx which
# can depend on stdlib modules removed in some Python versions (e.g. cgi).
Translator = None
spm = None

# 📂 Directory and File Paths
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "data.txt")
SCRAPED_FILE = os.path.join(DATA_DIR, "scraped_data.txt")
DICTIONARY_FILE = os.path.join(DATA_DIR, "dictionary.txt")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json")

# 📜 Logging Setup
LOG_FILE = "data_scraping.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_directories():
    """Creates necessary directories if they do not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info("📁 Data directory setup complete.")

def _ensure_spm_import():
    global spm
    if spm is None:
        try:
            import sentencepiece as spm_mod
            spm = spm_mod
        except Exception:
            spm = None

def train_sentencepiece_model(corpus_text: str):
    """Train a SentencePiece model from raw corpus text if enabled in config.

    Writes .model/.vocab to SPM_MODEL_PREFIX. Returns True on success.
    """
    try:
        from config import config as cfg
    except Exception:
        return False
    if not getattr(cfg, 'USE_SENTENCEPIECE', False):
        return False
    _ensure_spm_import()
    if spm is None:
        logging.warning("SentencePiece not available; skipping SPM training")
        return False
    prefix = getattr(cfg, 'SPM_MODEL_PREFIX', 'data/spm_lm')
    vocab_size = int(getattr(cfg, 'SPM_VOCAB_SIZE', 8000))
    char_cov = float(getattr(cfg, 'SPM_CHARACTER_COVERAGE', 0.9995))
    model_type = str(getattr(cfg, 'SPM_MODEL_TYPE', 'unigram'))
    uds = getattr(cfg, 'SPM_USER_DEFINED_SYMBOLS', ["<bos>","<eos>","<sep>","<nl>"])
    # Prepare a temporary file for training
    temp_corpus = os.path.join(DATA_DIR, 'spm_corpus.txt')
    try:
        with open(temp_corpus, 'w', encoding='utf-8') as tf:
            tf.write(corpus_text)
        spm.SentencePieceTrainer.Train(
            input=temp_corpus,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=char_cov,
            user_defined_symbols=uds,
        )
        logging.info(f"✅ Trained SentencePiece model at {prefix}.model")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to train SentencePiece: {e}")
        return False
    finally:
        try:
            os.remove(temp_corpus)
        except Exception:
            pass

def _load_spm_processor():
    try:
        from config import config as cfg
    except Exception:
        return None
    if not getattr(cfg, 'USE_SENTENCEPIECE', False):
        return None
    _ensure_spm_import()
    if spm is None:
        return None
    prefix = getattr(cfg, 'SPM_MODEL_PREFIX', 'data/spm_lm')
    model_path = prefix + '.model'
    if not os.path.exists(model_path):
        return None
    try:
        proc = spm.SentencePieceProcessor()
        proc.load(model_path)
        return proc
    except Exception:
        return None

def load_data(filepath):
    """Loads text data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"❌ File not found: {filepath}")
        return ""

def preprocess_data(text):
    """Language-model-friendly tokenization.

    - Lowercase (for a small corpus) to reduce sparsity
    - Preserve core punctuation tokens if configured
    - Split into sentences and wrap with BOS/EOS tokens (config-controlled)
    - Convert explicit newlines to a <nl> token when configured
    """
    from config import config as cfg
    # If SentencePiece enabled and model exists, use it directly to produce subword pieces.
    sp = _load_spm_processor()
    if sp is not None:
        # Directly return pieces (strings) so downstream uses the same pipeline
        pieces = sp.encode(text, out_type=str)
        return pieces
    t = text
    # Normalize newlines
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    # Optionally map newlines to token placeholders
    nl_tok = getattr(cfg, 'NEWLINE_TOKEN', None)
    if nl_tok:
        t = t.replace('\n\n', f' {nl_tok} ').replace('\n', f' {nl_tok} ')
    t = t.lower()

    bos = getattr(cfg, 'BOS_TOKEN', None)
    eos = getattr(cfg, 'EOS_TOKEN', None)
    add_wrappers = bool(getattr(cfg, 'INSERT_BOS_EOS', False))
    keep_punct = bool(getattr(cfg, 'INCLUDE_PUNCT_TOKENS', True))

    # Sentence split retaining end punctuation
    parts = re.split(r'([.!?])', t)
    sentences = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        end = parts[i+1] if i+1 < len(parts) else ''
        if seg:
            sentences.append(seg + end)

    tokens = []
    for s in sentences:
        # Tokenize words and (optionally) punctuation
        if keep_punct:
            # Split on spaces while isolating .,!?;:
            s = re.sub(r'([.,!?;:])', r' \1 ', s)
            toks = s.split()
        else:
            toks = re.findall(r"\b\w+\b", s)
        if not toks:
            continue
        if add_wrappers and bos:
            tokens.append(bos)
        tokens.extend(toks)
        if add_wrappers and eos:
            tokens.append(eos)
    # Fallback: if no sentence split happened, just basic tokenize
    if not tokens:
        if keep_punct:
            t2 = re.sub(r'([.,!?;:])', r' \1 ', t)
            tokens = t2.split()
        else:
            tokens = re.findall(r"\b\w+\b", t)
        if add_wrappers and eos:
            tokens.append(eos)
    return tokens

def build_vocabulary(tokens, min_freq=1):
    """Builds a vocabulary from tokens, filtering out low-frequency words.
    If the resulting vocab is too small, automatically relax min_freq.
    """
    # If SentencePiece is active, we assume fixed vocab loaded from .model and skip building.
    try:
        from config import config as cfg
        sp = _load_spm_processor()
        if getattr(cfg, 'USE_SENTENCEPIECE', False) and sp is not None:
            # Build vocab mapping from SPM ids to pieces; include PAD and UNK at 0/1 consistent with codebase
            word_to_index = {'<PAD>': 0, '<UNK>': 1}
            index_to_word = {0: '<PAD>', 1: '<UNK>'}
            size = sp.get_piece_size()
            idx = 2
            for i in range(size):
                piece = sp.id_to_piece(i)
                if piece not in word_to_index:
                    word_to_index[piece] = idx
                    index_to_word[idx] = piece
                    idx += 1
            # Ensure special tokens exist
            for special in (getattr(cfg,'BOS_TOKEN',None), getattr(cfg,'EOS_TOKEN',None), getattr(cfg,'SEP_TOKEN',None), getattr(cfg,'NEWLINE_TOKEN',None)):
                if special and special not in word_to_index:
                    word_to_index[special] = idx
                    index_to_word[idx] = special
                    idx += 1
            # Save for reuse
            try:
                import json
                with open(VOCAB_FILE, 'w', encoding='utf-8') as vf:
                    json.dump({'word_to_index': word_to_index, 'index_to_word': index_to_word}, vf, ensure_ascii=False)
            except Exception:
                pass
            return word_to_index, index_to_word
    except Exception:
        pass

    token_counts = Counter(tokens)
    filtered_tokens = [t for t, c in token_counts.items() if c >= min_freq]
    # Auto-relax if vocab would be tiny
    if len(filtered_tokens) < 100 and min_freq > 1:
        filtered_tokens = [t for t, c in token_counts.items() if c >= 1]

    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    index_to_word = {0: '<PAD>', 1: '<UNK>'}

    idx = 2
    bos_tok = None
    eos_tok = None
    sep_tok = None
    nl_tok = None
    try:
        from config import config as cfg
        bos_tok = getattr(cfg, 'BOS_TOKEN', None)
        eos_tok = getattr(cfg, 'EOS_TOKEN', None)
        sep_tok = getattr(cfg, 'SEP_TOKEN', None)
        nl_tok = getattr(cfg, 'NEWLINE_TOKEN', None)
    except Exception:
        bos_tok = eos_tok = sep_tok = nl_tok = None
    for token in filtered_tokens:
        if token not in word_to_index:
            word_to_index[token] = idx
            index_to_word[idx] = token
            idx += 1
    # Force add special tokens if configured and missing
    for special in (bos_tok, eos_tok, sep_tok, nl_tok):
        if special and special not in word_to_index:
            word_to_index[special] = idx
            index_to_word[idx] = special
            idx += 1
    # Save vocab for reuse
    try:
        import json
        with open(VOCAB_FILE, 'w', encoding='utf-8') as vf:
            json.dump({'word_to_index': word_to_index, 'index_to_word': index_to_word}, vf, ensure_ascii=False)
        logging.info("📖 Vocab saved.")
    except Exception:
        pass
    return word_to_index, index_to_word

def try_load_vocab():
    """Load saved vocabulary if available."""
    try:
        import json
        if os.path.exists(VOCAB_FILE):
            with open(VOCAB_FILE, 'r', encoding='utf-8') as vf:
                data = json.load(vf)
                w2i = data.get('word_to_index', None)
                if isinstance(w2i, dict) and '<PAD>' in w2i and '<UNK>' in w2i:
                    i2w = {int(v): k for k, v in w2i.items()}
                    return w2i, i2w
    except Exception:
        pass
    return None, None

def tokenize_and_numericalize(tokens, word_to_index):
    """Converts tokens to numerical indices based on vocabulary."""
    return [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens]

def create_sequences(numericalized_tokens, sequence_length):
    """Creates overlapping sequences from numericalized tokens."""
    return [numericalized_tokens[i:i + sequence_length + 1]
            for i in range(len(numericalized_tokens) - sequence_length)]

def split_data(sequences, split_ratio=0.8):
    """Splits data into training and validation sets."""
    import random
    random.shuffle(sequences)
    split_idx = int(len(sequences) * split_ratio)
    return sequences[:split_idx], sequences[split_idx:]

class TextDataset(Dataset):
    """PyTorch Dataset for text sequences."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        # Next-token prediction across the full sequence window:
        # inputs:  [w0, w1, ..., w_{S-1}]
        # targets: [w1, w2, ..., w_S]
        inputs = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        return inputs, targets

def data_to_tensors(sequences, device):
    """Converts sequences into PyTorch tensors."""
    inputs = torch.tensor([seq[:-1] for seq in sequences], dtype=torch.long).to(device)
    targets = torch.tensor([seq[-1] for seq in sequences], dtype=torch.long).to(device)
    return inputs, targets

def fetch_wikipedia_articles(topics, lang="en"):
    """Scrapes Wikipedia pages for given topics.

    The wikipediaapi package is optional. If it's not available we log a
    warning and skip scraping.
    """
    try:
        import wikipediaapi
    except Exception:
        logging.warning("wikipediaapi not available -- skipping wikipedia scraping")
        return

    # wikipediaapi doesn't always accept a user_agent parameter in all versions.
    # We'll try to construct it with a user_agent when available; otherwise
    # we'll fall back to setting the underlying session's headers if possible.
    try:
        from config import config as cfg
        default_ua = getattr(cfg, 'USER_AGENT', None)
    except Exception:
        default_ua = None

    try:
        # Preferred call: pass user_agent as named arg (newer wikipediaapi versions)
        if default_ua:
            wiki = wikipediaapi.Wikipedia(language=lang, user_agent=default_ua)
        else:
            wiki = wikipediaapi.Wikipedia(language=lang)
    except TypeError:
        # Fallback: try without user_agent then set session headers if exposed
        wiki = wikipediaapi.Wikipedia(lang)
        try:
            # Some versions expose a session or http attribute where headers can be set
            if default_ua and hasattr(wiki, 'session'):
                wiki.session.headers.update({'User-Agent': default_ua})
        except Exception:
            # Give up quietly and continue — the caller will see Wikipedia's own errors
            pass
    with open(SCRAPED_FILE, "a", encoding="utf-8") as file:
        for topic in topics:
            try:
                page = wiki.page(topic)
                if page.exists():
                    text = clean_text(page.text)
                    file.write(text + "\n\n")
                    logging.info(f"✅ Scraped Wikipedia: {topic} ({lang})")
                else:
                    logging.warning(f"⚠️ Wikipedia page not found: {topic}")
            except Exception as e:
                logging.error(f"❌ Error scraping Wikipedia page {topic}: {e}")

def fetch_live_data(urls):
    """Scrapes data from given URLs."""
    with open(SCRAPED_FILE, "a", encoding="utf-8") as file:
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    text = clean_text("\n".join([p.get_text() for p in paragraphs]))
                    file.write(text + "\n\n")
                    logging.info(f"✅ Scraped: {url}")
                else:
                    logging.warning(f"⚠️ Failed to fetch {url} (Status: {response.status_code})")
            except Exception as e:
                logging.error(f"❌ Error scraping {url}: {e}")

def clean_text(text):
    """Cleans text by removing extra spaces and special characters."""
    return re.sub(r'\s+', ' ', text).strip()

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer when detected."""
    # Headers often between '*** START' and '*** END' markers
    start = re.search(r'\*\*\*\s*START[^\n]*\n', text, re.IGNORECASE)
    end = re.search(r'\*\*\*\s*END[^\n]*\n', text, re.IGNORECASE)
    if start and end and end.start() > start.end():
        return text[start.end():end.start()]
    return text

def _normalize_quotes(text: str) -> str:
    # Normalize curly quotes to straight quotes for consistency
    return (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))

def _ascii_only(text: str) -> str:
    return ''.join(ch for ch in text if ord(ch) < 128)

def _split_sentences(text: str):
    # Rough sentence splitting by punctuation boundaries
    parts = re.split(r'([.!?])', text)
    out = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        end = parts[i+1] if i+1 < len(parts) else ''
        if seg:
            out.append(seg + end)
    return out

def _alpha_ratio(s: str) -> float:
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(1, len(s))

def clean_english_corpus(text: str) -> str:
    """Apply stricter English cleaning: normalize quotes, dedupe sentences,
    filter sentences by alpha-ratio and length, and collapse whitespace.
    Controlled by config flags in config/config.py
    """
    try:
        from config import config as cfg
    except Exception:
        return clean_text(text)
    if not getattr(cfg, 'CLEAN_ENGLISH', False):
        return clean_text(text)
    t = text
    if getattr(cfg, 'CLEAN_NORMALIZE_QUOTES', True):
        t = _normalize_quotes(t)
    if getattr(cfg, 'CLEAN_STRIP_NON_ASCII', False):
        t = _ascii_only(t)
    sents = _split_sentences(t)
    dedup = set()
    kept = []
    minc = int(getattr(cfg, 'CLEAN_MIN_CHARS_PER_SENT', 20))
    maxc = int(getattr(cfg, 'CLEAN_MAX_CHARS_PER_SENT', 220))
    min_alpha = float(getattr(cfg, 'CLEAN_MIN_ALPHA_RATIO', 0.85))
    for s in sents:
        s0 = s.strip()
        if len(s0) < minc or len(s0) > maxc:
            continue
        letters = sum(ch.isalpha() for ch in s0)
        ratio = letters / max(1, len(s0))
        if ratio < min_alpha:
            continue
        key = s0.lower()
        if getattr(cfg, 'CLEAN_DEDUP', True):
            if key in dedup:
                continue
            dedup.add(key)
        kept.append(s0)
    return clean_text(' '.join(kept))

def translate_text(text, target_languages):
    """Translates text into multiple languages using multi-threading.

    The googletrans package is optional. If it's not available or fails
    to import, this function returns an empty dict and logs a warning.
    """
    try:
        # lazy import to avoid importing httpx at module import time
        from googletrans import Translator as _Translator
    except Exception:
        logging.warning("googletrans not available or failed to import -- skipping translations")
        return {}

    try:
        translator = _Translator()
    except Exception:
        logging.error("Failed to initialize Translator")
        return {}

    translations = {}

    def translate(lang):
        try:
            translations[lang] = translator.translate(text, dest=lang).text
            logging.info(f"🌍 Translated text to {lang}")
        except Exception as e:
            logging.error(f"❌ Translation error for {lang}: {e}")

    with ThreadPoolExecutor() as executor:
        executor.map(translate, target_languages)

    return translations

def compile_data():
    """Merges all text sources into 'data.txt' and optional JSONL for LM-friendly training."""
    try:
        from config import config as cfg
    except Exception:
        class _C:
            pass
        cfg = _C()
    produce_jsonl = bool(getattr(cfg, 'DATA_JSONL', True))
    jsonl_path = getattr(cfg, 'JSONL_PATH', os.path.join(DATA_DIR, 'data.jsonl'))
    strip_headers = bool(getattr(cfg, 'GUTENBERG_STRIP_HEADERS', True))

    with open(DATA_FILE, "w", encoding="utf-8") as output_file:
        jsonl_f = open(jsonl_path, 'w', encoding='utf-8') if produce_jsonl else None
        try:
            for filename in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, filename)
                if filename.endswith(".txt") and filename not in ["data.txt"]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            raw = f.read()
                            if strip_headers:
                                raw = strip_gutenberg_boilerplate(raw)
                            cleaned = clean_english_corpus(raw)
                            output_file.write(cleaned + "\n\n")
                            if jsonl_f is not None:
                                # Sentence-level JSONL for easier LM training
                                sents = _split_sentences(cleaned)
                                # Filter tiny sources
                                min_sents = int(getattr(cfg, 'MIN_SENTENCES_PER_SOURCE', 0))
                                if len(sents) < min_sents:
                                    logging.info(f"⏭️  Skipping {filename} due to low sentence count ({len(sents)}<{min_sents})")
                                else:
                                    import json
                                    for idx, s in enumerate(sents):
                                        s0 = s.strip()
                                        if not s0:
                                            continue
                                        rec = {
                                            "text": s0,
                                            "source": filename,
                                            "index": idx,
                                            "length": len(s0),
                                            "alpha_ratio": _alpha_ratio(s0)
                                        }
                                        jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        logging.info(f"✅ Merged: {filename}")
                    except Exception as e:
                        logging.error(f"❌ Error reading {filename}: {e}")
        finally:
            if jsonl_f is not None:
                jsonl_f.close()

    # Optional JSONL validation pass
    if produce_jsonl and bool(getattr(cfg, 'JSONL_VALIDATE', True)):
        try:
            import json
            bad = 0
            total = 0
            with open(jsonl_path, 'r', encoding='utf-8') as jf:
                for line in jf:
                    total += 1
                    try:
                        rec = json.loads(line)
                        if not rec.get('text'):
                            bad += 1
                    except Exception:
                        bad += 1
            if bad > 0:
                logging.warning(f"⚠️ JSONL validation: {bad}/{total} invalid lines in {jsonl_path}")
            else:
                logging.info(f"✅ JSONL validation passed for {jsonl_path} ({total} lines)")
        except Exception as e:
            logging.error(f"❌ JSONL validation error: {e}")

    # Optional JSONL meta summary
    if produce_jsonl and bool(getattr(cfg, 'JSONL_WRITE_META', True)):
        meta_path = getattr(cfg, 'JSONL_META_PATH', os.path.join(DATA_DIR, 'data_meta.json'))
        try:
            import json
            total = 0
            total_len = 0
            total_alpha = 0.0
            per_source = {}
            with open(jsonl_path, 'r', encoding='utf-8') as jf:
                for line in jf:
                    rec = json.loads(line)
                    s = rec.get('text', '')
                    src = rec.get('source', 'unknown')
                    total += 1
                    total_len += len(s)
                    total_alpha += _alpha_ratio(s)
                    per_source[src] = per_source.get(src, 0) + 1
            avg_len = total_len / max(1, total)
            avg_alpha = total_alpha / max(1, total)
            meta = {
                'total_lines': total,
                'avg_length': avg_len,
                'avg_alpha_ratio': avg_alpha,
                'per_source': per_source,
                'jsonl_path': jsonl_path
            }
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
            logging.info(f"🧾 Wrote JSONL meta summary to {meta_path}")
        except Exception as e:
            logging.error(f"❌ JSONL meta summary error: {e}")

def build_jsonl_from_data_file(jsonl_path=None):
    """Fallback: build a sentence-level JSONL from DATA_FILE when sources are unavailable.

    Writes records with fields: text, source='data.txt', index, length, alpha_ratio
    """
    try:
        from config import config as cfg
    except Exception:
        class _C:
            pass
        cfg = _C()
    if jsonl_path is None:
        jsonl_path = getattr(cfg, 'JSONL_PATH', os.path.join(DATA_DIR, 'data.jsonl'))
    txt = load_data(DATA_FILE)
    if not txt:
        logging.warning("⚠️ build_jsonl_from_data_file: data.txt is empty or missing")
        return
    cleaned = clean_english_corpus(txt)
    sents = _split_sentences(cleaned)
    import json
    with open(jsonl_path, 'w', encoding='utf-8') as jf:
        for idx, s in enumerate(sents):
            s0 = s.strip()
            if not s0:
                continue
            rec = {
                'text': s0,
                'source': 'data.txt',
                'index': idx,
                'length': len(s0),
                'alpha_ratio': _alpha_ratio(s0)
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + '\n')
    logging.info(f"📝 Built JSONL from data.txt -> {jsonl_path} ({len(sents)} sentences)")

    logging.info(f"📂 Merged data saved to '{DATA_FILE}'")

def build_dictionary(text):
    """Extracts unique words and saves a dictionary file."""
    words = sorted(set(re.findall(r'\b\w+\b', text.lower())))
    with open(DICTIONARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logging.info("📖 Dictionary saved.")

if __name__ == "__main__":
    setup_directories()

    wiki_topics = ["Greetings", "Etiquete", "Response to questions", "Type of questions","simple greatings"]
    urls_to_scrape = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "https://www.gutenberg.org/files/11/11-0.txt"
    ]

    print("🔄 Fetching Wikipedia data...")
    fetch_wikipedia_articles(wiki_topics)

    print("🔄 Fetching custom website data...")
    fetch_live_data(urls_to_scrape)

    print("🔄 Compiling data...")
    compile_data()

    text_data = load_data(DATA_FILE)
    build_dictionary(text_data)

    sample_text = "Artificial Intelligence is transforming the world."
    translations = translate_text(sample_text, ["fr", "es", "de", "sw"])
    print("🌍 Translations:", translations)

    print("✅ Data processing complete!")

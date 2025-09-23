import os
import re
import logging
import torch
import requests
from collections import Counter
# ThreadPoolExecutor is a stdlib import; keep imports grouped at the top
from concurrent.futures import ThreadPoolExecutor
# BeautifulSoup is required; wikipediaapi is optional and imported lazily
from bs4 import BeautifulSoup
# googletrans import is intentionally lazy inside translate_text()
# because importing it at module import time pulls in httpx which
# can depend on stdlib modules removed in some Python versions (e.g. cgi).
Translator = None

# 📂 Directory and File Paths
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "data.txt")
SCRAPED_FILE = os.path.join(DATA_DIR, "scraped_data.txt")
DICTIONARY_FILE = os.path.join(DATA_DIR, "dictionary.txt")

# 📜 Logging Setup
LOG_FILE = "data_scraping.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_directories():
    """Creates necessary directories if they do not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.info("📁 Data directory setup complete.")

def load_data(filepath):
    """Loads text data from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"❌ File not found: {filepath}")
        return ""

def preprocess_data(text):
    """Lowercases and tokenizes text, removing punctuation."""
    text = text.lower()
    tokens = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split()
    return tokens

def build_vocabulary(tokens, min_freq=1):
    """Builds a vocabulary from tokens, filtering out low-frequency words."""
    token_counts = Counter(tokens)
    filtered_tokens = [t for t, c in token_counts.items() if c >= min_freq]

    word_to_index = {'<PAD>': 0, '<UNK>': 1}
    index_to_word = {0: '<PAD>', 1: '<UNK>'}

    for i, token in enumerate(filtered_tokens, start=2):
        word_to_index[token] = i
        index_to_word[i] = token

    return word_to_index, index_to_word

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
    """Merges all text sources into 'data.txt'."""
    with open(DATA_FILE, "w", encoding="utf-8") as output_file:
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if filename.endswith(".txt") and filename not in ["data.txt"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        output_file.write(clean_text(f.read()) + "\n\n")
                    logging.info(f"✅ Merged: {filename}")
                except Exception as e:
                    logging.error(f"❌ Error reading {filename}: {e}")

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

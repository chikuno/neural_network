# data/augmentation.py
import random
import logging
import torch

# Do NOT download WordNet at import time (blocks startup). We'll try to
# use WordNet lazily inside synonym_replacement and fall back if it's
# unavailable.

def synonym_replacement(tokens, n=2):
    """
    Replaces up to n words in the token list with synonyms from WordNet.
    Only replaces words longer than 3 characters.
    """
    # Attempt to use NLTK WordNet; if not available, skip synonym replacement.
    try:
        from nltk.corpus import wordnet
    except Exception:
        logging.warning("NLTK WordNet not available; skipping synonym replacement")
        return tokens

    new_tokens = tokens.copy()
    candidates = [word for word in tokens if len(word) > 3]
    random.shuffle(candidates)
    num_replaced = 0
    for word in candidates:
        try:
            synsets = wordnet.synsets(word)
        except Exception:
            synsets = []
        if synsets:
            synonym = synsets[0].lemmas()[0].name().replace('_', ' ')
            # Replace only the first occurrence of the word.
            new_tokens[new_tokens.index(word)] = synonym
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_tokens

def back_translate(text, lang='fr'):
    """
    Translates text to another language and then back to English for augmentation.
    """
    try:
        # Import lazily so missing optional dependency doesn't break imports
        from deep_translator import GoogleTranslator
    except Exception:
        logging.warning("deep_translator not available; skipping back-translation augmentation")
        return text

    try:
        translator = GoogleTranslator(source='auto', target=lang)
        translated = translator.translate(text)
        back_translator = GoogleTranslator(source=lang, target='en')
        back_translated = back_translator.translate(translated)
        return back_translated
    except Exception as e:
        logging.warning(f"Error in back translation: {e} -- returning original text")
        return text

def augment_data(tokens):
    """
    Applies augmentation techniques to the token list.
    Returns a list of augmented token lists.
    """
    # If tokens is a torch Tensor (numericalized sequences), convert to list(s)
    if isinstance(tokens, torch.Tensor):
        # Convert tensor of sequences (batch x seq_len) into list of lists
        try:
            sequences = tokens.tolist()
        except Exception:
            # If it's a single sequence tensor, make it a list
            sequences = [tokens.cpu().numpy().tolist()]
        # For numericalized sequences we cannot perform WordNet-based or
        # back-translation text operations here (we don't have the mapping to words),
        # so return the sequences unchanged as a single-element list of sequences.
        return sequences

    augmented_versions = []
    # Option 1: Synonym Replacement
    if random.random() > 0.5:
        augmented_versions.append(synonym_replacement(tokens, n=2))
    # Option 2: Back Translation
    if random.random() > 0.5:
        original_text = " ".join(tokens)
        back_translated = back_translate(original_text)
        augmented_versions.append(back_translated.split())
    return augmented_versions if augmented_versions else [tokens]

# data/augmentation.py
import random
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator

# Ensure WordNet is downloaded
nltk.download('wordnet')

def synonym_replacement(tokens, n=2):
    """
    Replaces up to n words in the token list with synonyms from WordNet.
    Only replaces words longer than 3 characters.
    """
    new_tokens = tokens.copy()
    candidates = [word for word in tokens if len(word) > 3]
    random.shuffle(candidates)
    num_replaced = 0
    for word in candidates:
        synsets = wordnet.synsets(word)
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
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        back_translated = GoogleTranslator(source=lang, target='en').translate(translated)
        return back_translated
    except Exception as e:
        print("Error in back translation:", e)
        return text

def augment_data(tokens):
    """
    Applies augmentation techniques to the token list.
    Returns a list of augmented token lists.
    """
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

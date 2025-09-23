import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import LSTMTextGenerationModel
from inference.generate import generate_text_simple
from config import config

# Build a tiny vocabulary
word_to_index = {'<PAD>':0, '<UNK>':1, 'the':2, 'quick':3, 'brown':4, 'fox':5, 'jumps':6, 'over':7, 'lazy':8, 'dog':9}
index_to_word = {i:w for w,i in word_to_index.items()}

vocab_size = len(word_to_index)

# Create a small LSTM model (randomly initialized)
model = LSTMTextGenerationModel(vocab_size, config.EMBEDDING_DIM, 64, 1, config.DROPOUT)

out = generate_text_simple(model, ['the', 'quick', 'brown'], word_to_index, index_to_word, 20, 1.0, 'cpu', model_type='lstm')
print('Generated:', out)

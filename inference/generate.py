# inference/generate.py

import torch
import torch.nn.functional as F

def generate_text_simple(model, start_words, word_to_index, index_to_word, num_words, temperature, device, model_type):
    """
    Generates text using greedy sampling.
    
    Args:
        model: Trained model.
        start_words: List of seed words.
        word_to_index: Vocabulary mapping.
        index_to_word: Reverse vocabulary mapping.
        num_words: Number of words to generate.
        temperature: Sampling temperature.
        device: Computation device.
        model_type: Model type ('rnn', 'gru', 'lstm', or 'transformer').
        
    Returns:
        Generated text string.
    """
    model.eval()
    with torch.no_grad():
        input_sequence = [word_to_index.get(word, word_to_index['<UNK>']) for word in start_words]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
        generated_words = start_words.copy()
        for _ in range(num_words):
            if model_type in ['rnn', 'gru', 'lstm']:
                hidden = model.init_hidden(1, device)
                logits, _, hidden = model(input_tensor, hidden)
            else:
                logits, _ = model(input_tensor)
            probabilities = F.softmax(logits / temperature, dim=1)
            predicted_index = torch.multinomial(probabilities, 1).item()
            generated_words.append(index_to_word.get(predicted_index, '<UNK>'))
            input_tensor = torch.tensor([predicted_index], dtype=torch.long).unsqueeze(0).to(device)
        return " ".join(generated_words)

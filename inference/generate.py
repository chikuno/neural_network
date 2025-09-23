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
        # Accept start_words as either:
        # - a string (space-separated words),
        # - a list of words, or
        # - a list of token ids (ints) produced by the preprocessing pipeline.
        if isinstance(start_words, str):
            start_list = start_words.split()
            token_mode = False
        else:
            start_list = list(start_words)
            # If every element is an int, treat as token id list
            token_mode = all(isinstance(x, int) for x in start_list)

        # Build initial input sequence indices and generated word buffer
        if token_mode:
            input_sequence = start_list
            generated_words = [index_to_word.get(int(i), '<UNK>') for i in input_sequence]
        else:
            input_sequence = [word_to_index.get(word, word_to_index.get('<UNK>', 0)) for word in start_list]
            generated_words = list(start_list)

        # Helper to make a one-hot float tensor for MLP models (used only when model has no embedding)
        def one_hot_tensor(index, size):
            vec = torch.zeros(1, size, dtype=torch.float, device=device)
            if 0 <= index < size:
                vec[0, index] = 1.0
            return vec

        # Prepare the initial input tensor depending on model type
        if model_type == 'mlp' or (hasattr(model, 'input_size') and model_type == 'mlp'):
            # If model exposes an embedding, use it (pass index tensors and let the model embed)
            if hasattr(model, 'embed'):
                last_index = input_sequence[-1] if len(input_sequence) > 0 else word_to_index.get('<UNK>', 0)
                input_tensor = torch.tensor([last_index], dtype=torch.long).to(device)
            else:
                vocab_size = getattr(model, 'input_size', None)
                if vocab_size is None:
                    # Fallback: use max index + 1 from word_to_index
                    vocab_size = max(word_to_index.values()) + 1
                last_index = input_sequence[-1] if len(input_sequence) > 0 else word_to_index.get('<UNK>', 0)
                input_tensor = one_hot_tensor(last_index, vocab_size)
        else:
            input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(num_words):
            if model_type in ['rnn', 'gru', 'lstm']:
                hidden = model.init_hidden(1, device)
                logits, _, hidden = model(input_tensor, hidden)
            elif model_type == 'mlp':
                # MLP expects float inputs (e.g., one-hot or embedded vectors)
                logits = model(input_tensor)
            else:
                # Transformer or other models that accept token indices
                # Ensure indices are long dtype
                if input_tensor.dtype != torch.long:
                    input_tensor = input_tensor.to(torch.long)
                logits, _ = model(input_tensor)

            # If logits has extra dimensions (e.g., sequence dim), reduce to last token
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            probabilities = F.softmax(logits / max(temperature, 1e-8), dim=1)
            predicted_index = torch.multinomial(probabilities, 1).item()
            generated_words.append(index_to_word.get(predicted_index, '<UNK>'))

            # Prepare next input tensor according to model type
            if model_type == 'mlp':
                if hasattr(model, 'embed'):
                    input_tensor = torch.tensor([predicted_index], dtype=torch.long).to(device)
                else:
                    # fallback to one-hot if no embedding
                    vocab_sz = getattr(model, 'input_size', None)
                    if vocab_sz is None:
                        vocab_sz = max(word_to_index.values()) + 1
                    input_tensor = one_hot_tensor(predicted_index, vocab_sz)
            else:
                input_tensor = torch.tensor([predicted_index], dtype=torch.long).unsqueeze(0).to(device)

        # Post-process generated words: remove placeholders and collapse repeats
        cleaned = []
        placeholders = {'<PAD>', '<UNK>'}
        for w in generated_words:
            if w in placeholders:
                continue
            if len(cleaned) > 0 and cleaned[-1] == w:
                continue
            cleaned.append(w)
        return " ".join(cleaned)

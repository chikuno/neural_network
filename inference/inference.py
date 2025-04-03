# inference.py

import torch
import torch.nn.functional as F

def beam_search(model, start_words, word_to_index, index_to_word, beam_width=3,
                max_length=100, temperature=1.0, device="cpu", model_type='lstm'):
    """
    Generate text using beam search.
    Returns the generated sequence with the highest cumulative log probability.
    """
    model.eval()
    with torch.no_grad():
        # Initialize beam with start sequence
        init_seq = [word_to_index.get(word, word_to_index['<UNK>']) for word in start_words]
        beams = [(init_seq, 0.0, None)]  # (sequence, cumulative log-prob, hidden state)

        for _ in range(max_length):
            new_beams = []
            for seq, cum_log_prob, hidden in beams:
                input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                if model_type in ['rnn', 'gru', 'lstm']:
                    if hidden is None:
                        hidden = model.init_hidden(1, device)
                    logits, _, hidden_out = model(input_tensor, hidden)
                else:
                    logits, _ = model(input_tensor)
                    hidden_out = None

                logits = logits.squeeze(0) / temperature
                probs = F.log_softmax(logits, dim=0)
                top_log_probs, top_indices = torch.topk(probs, beam_width)
                for log_prob, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_seq = seq + [idx]
                    new_cum_log_prob = cum_log_prob + log_prob
                    new_beams.append((new_seq, new_cum_log_prob, hidden_out))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq, best_score, _ = beams[0]
        generated_words = [index_to_word[idx] for idx in best_seq]
        return ' '.join(generated_words)

def generate_text_simple(model, start_words, word_to_index, index_to_word,
                         num_words_to_generate, temperature=1.0, device="cpu", model_type='lstm'):
    """
    Generates text using simple (greedy) sampling.
    """
    model.eval()
    with torch.no_grad():
        input_sequence = [word_to_index.get(word, word_to_index['<UNK>']) for word in start_words]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
        generated_text = start_words[:]
        if model_type in ['rnn', 'gru', 'lstm']:
            hidden = model.init_hidden(1, device)
            for _ in range(num_words_to_generate):
                logits, _, hidden = model(input_tensor, hidden)
                probabilities = torch.softmax(logits / temperature, dim=1)
                predicted_index = torch.multinomial(probabilities, 1).item()
                generated_text.append(index_to_word[predicted_index])
                input_tensor = torch.tensor([predicted_index], dtype=torch.long).unsqueeze(0).to(device)
        else:
            for _ in range(num_words_to_generate):
                logits, _ = model(input_tensor)
                probabilities = torch.softmax(logits / temperature, dim=1)
                predicted_index = torch.multinomial(probabilities, 1).item()
                generated_text.append(index_to_word[predicted_index])
                new_seq = input_tensor.squeeze(0).tolist() + [predicted_index]
                input_tensor = torch.tensor(new_seq[-input_tensor.size(1):], dtype=torch.long).unsqueeze(0).to(device)
    return ' '.join(generated_text)

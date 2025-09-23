# data/active_learning.py
import torch
import torch.nn.functional as F

def calculate_entropy(predictions):
    """Calculates entropy for uncertainty estimation."""
    return -torch.sum(predictions * torch.log(predictions + 1e-9), dim=1)

def select_uncertain_samples(model, data_inputs, top_k=10, device='cpu'):
    """
    Selects the top_k most uncertain samples based on entropy.
    
    Args:
        model: Trained model.
        data_inputs: Input tensor.
        top_k: Number of uncertain samples to select.
        device: Device for computation.
        
    Returns:
        A tensor containing the most uncertain samples.
    """
    model.eval()
    # Handle empty input
    if data_inputs is None or data_inputs.numel() == 0:
        return data_inputs

    # Ensure batch dimension: if provided a single sample (1D), make it batch size 1
    if data_inputs.dim() == 1:
        data_inputs = data_inputs.unsqueeze(0)

    with torch.no_grad():
        try:
            outputs = model(data_inputs.to(device))
        except TypeError:
            # Some models (RNN/GRU/LSTM) expect a hidden state as second arg.
            batch_size = data_inputs.size(0)
            hidden = model.init_hidden(batch_size, device)
            outputs = model(data_inputs.to(device), hidden)

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        entropy_scores = calculate_entropy(probabilities)
    
    uncertain_indices = torch.argsort(entropy_scores, descending=True)[:top_k]
    return data_inputs[uncertain_indices]

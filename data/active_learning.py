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
    with torch.no_grad():
        outputs = model(data_inputs.to(device))
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = F.softmax(logits, dim=1)
        entropy_scores = calculate_entropy(probabilities)
    
    uncertain_indices = torch.argsort(entropy_scores, descending=True)[:top_k]
    return data_inputs[uncertain_indices]

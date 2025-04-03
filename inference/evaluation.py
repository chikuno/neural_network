# inference/evaluation.py

import torch
import torch.nn.functional as F

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataloader.
    
    Args:
        model: Trained model.
        dataloader: DataLoader for the validation/test set.
        criterion: Loss function.
        device: Computation device.
        
    Returns:
        Average loss over the dataloader.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

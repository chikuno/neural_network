# utils.py

import torch
import matplotlib.pyplot as plt
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Saves the model checkpoint."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, checkpoint_path)
    print(f"✅ Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """Loads the model checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
        return checkpoint
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}")
        return None

def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    """Plots and saves training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', marker='s', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curves saved to {save_path}")

def weighted_ensemble_output(ensemble_outputs):
    """
    Combines outputs from multiple ensemble models.

    Args:
        ensemble_outputs (list of tuples): Each tuple contains (generated_text, weight).
    
    Returns:
        str: The generated text with the highest weight.
    """
    if not ensemble_outputs:
        return None
    return max(ensemble_outputs, key=lambda x: x[1])[0]

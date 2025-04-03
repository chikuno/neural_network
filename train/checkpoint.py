# train/checkpoint.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Saves model and optimizer state to a checkpoint file."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """Loads model and optimizer state from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
        return checkpoint
    else:
        print("No checkpoint found at", checkpoint_path)
        return None

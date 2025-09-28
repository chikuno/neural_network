import torch
from typing import Iterable, Optional

def compute_class_weights(train_sequences: Iterable[Iterable[int]],
                          val_sequences: Iterable[Iterable[int]],
                          vocab_size: int,
                          power: float = 0.5,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute inverse-frequency class weights across all tokens.

    Args:
        train_sequences: iterable of sequences (list/tuple of ints)
        val_sequences: iterable of sequences
        vocab_size: total vocabulary size (ensures weight length)
        power: exponent applied to inverse frequency (0.5 -> sqrt)
        device: optional device for returned tensor

    Returns:
        1D tensor of shape (vocab_size,) with normalized weights summing to vocab_size.
    """
    try:
        # Flatten tokens
        train_tokens = [tok for seq in train_sequences for tok in seq]
        val_tokens = [tok for seq in val_sequences for tok in seq]
        flat = torch.tensor(train_tokens + val_tokens, dtype=torch.long)
    except Exception:
        flat = torch.empty(0, dtype=torch.long)

    if flat.numel() > 0:
        counts = torch.bincount(flat, minlength=vocab_size).float()
    else:
        counts = torch.ones(vocab_size, dtype=torch.float)

    if counts.numel() != vocab_size:
        if counts.numel() < vocab_size:
            counts = torch.cat([counts, torch.ones(vocab_size - counts.numel())], dim=0)
        else:
            counts = counts[:vocab_size]

    counts[counts == 0] = 1.0
    inv = (1.0 / counts) ** float(power)
    inv = inv / inv.sum() * len(inv)
    if device is not None:
        inv = inv.to(device)
    return inv

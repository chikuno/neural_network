import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_layers, output_size, dropout=0.5):
        super(MLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        layers = []
        in_features = embedding_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        
        layers.append(nn.Linear(in_features, output_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x is expected to be a tensor of token indices of shape (batch_size, sequence_length)
        # However, this simple MLP will just take the last token as input.
        # A more complex MLP could flatten the embeddings of the whole sequence.
        last_token_indices = x[:, -1]
        embedded = self.embedding(last_token_indices)
        output = self.mlp(embedded)
        return output, None # Return None for aux_logits to match other models' signatures


class AdvancedMLP(nn.Module):
    """Flexible MLP used in tests.

    Features:
    - Optional embedding layer when `use_embedding=True` (expects 1D LongTensor of token indices)
    - Standard MLP with Linear -> BatchNorm1d -> ReLU -> Dropout blocks
    - Returns logits tensor of shape (batch, output_size)
    """

    def __init__(self,
                 input_size: int,
                 hidden_layers,
                 output_size: int,
                 use_embedding: bool = False,
                 vocab_size: int = None,
                 embedding_dim: int = None,
                 dropout: float = 0.5):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            if vocab_size is None or embedding_dim is None:
                raise ValueError("vocab_size and embedding_dim must be provided when use_embedding=True")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            effective_input = embedding_dim
        else:
            effective_input = input_size

        layers = []
        in_features = effective_input
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            # BatchNorm adds a tiny overhead but aligns with test comment about batch size >=2
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If using embedding, x expected shape: (batch,) of token indices
        if self.use_embedding:
            if x.dim() != 1:
                # Allow (batch, seq) -> take last token similar to simpler MLP
                x = x[:, -1]
            x = self.embedding(x)  # (batch, embedding_dim)
        else:
            # Expect (batch, features); if 1D, unsqueeze
            if x.dim() == 1:
                x = x.unsqueeze(0)
        return self.net(x)


__all__ = [name for name in globals().keys() if name in ("MLPModel", "AdvancedMLP")]


import torch
import torch.nn as nn
import numpy as np


# Transformer-based Model

class TransformerTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dropout=0.5, multi_task=False, tie_weights=True):
        super(TransformerTextGenerationModel, self).__init__()
        self.multi_task = multi_task
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # Optional embedding LayerNorm for stability at larger widths
        try:
            from config import config as cfg
            self.embed_norm = nn.LayerNorm(embedding_dim) if bool(getattr(cfg, 'EMBED_LAYER_NORM', False)) else None
            ff_mult = float(getattr(cfg, 'TRANSFORMER_FF_MULT', 4.0))
        except Exception:
            self.embed_norm = None
            ff_mult = 4.0
        # Pre-norm transformer layer tends to be more stable
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            dim_feedforward=int(round(ff_mult*embedding_dim))
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.tie_weights = tie_weights
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=not tie_weights)
        if tie_weights:
            self.linear.weight = self.embedding.weight
        if self.multi_task:
            self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        if self.embed_norm is not None:
            embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)
        # Batch-first PositionalEncoding compatibility
        embedded = self.pos_encoder(embedded, batch_first=True)
        # Causal (subsequent) mask to prevent peeking ahead; shape must be (S, S)
        if src_mask is None:
            seq_len = x.size(1)
            src_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        output = self.transformer_encoder(embedded, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)
        # Produce logits for each time step (batch, seq, vocab)
        logits = self.linear(output)
        if self.multi_task:
            # Use final step hidden for aux classification
            aux_logits = self.classifier(output[:, -1, :])
            return logits, aux_logits
        return logits, None



 #Positional Encoding for Transformer
 
class PositionalEncoding(nn.Module):
    """Implements the positional encoding as in 'Attention is All You Need'."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x, batch_first=False):
        # support both (seq, batch, embed) and (batch, seq, embed)
        if batch_first:
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)
            return self.dropout(x)
        else:
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        
          
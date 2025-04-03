import torch
import torch.nn as nn
import numpy as np


# Transformer-based Model

class TransformerTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dropout=0.5, multi_task=False):
        super(TransformerTextGenerationModel, self).__init__()
        self.multi_task = multi_task
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        if self.multi_task:
            self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, x, src_mask=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        # Transformer expects (seq_len, batch_size, embedding_dim)
        embedded = self.pos_encoder(embedded.transpose(0, 1))
        output = self.transformer_encoder(embedded, src_mask)
        output = output[-1, :, :]  # Use last time step for prediction
        logits = self.linear(output)
        if self.multi_task:
            aux_logits = self.classifier(output)
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

    def forward(self, x):
        # x: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
          
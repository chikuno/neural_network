import torch
import torch.nn as nn

######################################
# Vanilla RNN-based Model
######################################
class RNNTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, multi_task=False, tie_weights=True):
        super(RNNTextGenerationModel, self).__init__()
        self.multi_task = multi_task
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        try:
            from config import config as cfg
            self.embed_norm = nn.LayerNorm(embedding_dim) if bool(getattr(cfg, 'EMBED_LAYER_NORM', False)) else None
        except Exception:
            self.embed_norm = None
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.tie_weights = tie_weights
        # Decoder setup with optional weight tying
        decoder_in = hidden_size
        self.decoder_proj = None
        if tie_weights:
            if hidden_size != embedding_dim:
                self.decoder_proj = nn.Linear(hidden_size, embedding_dim, bias=False)
                decoder_in = embedding_dim
            else:
                decoder_in = hidden_size
            self.linear = nn.Linear(decoder_in, vocab_size, bias=False)
            # Tie weights
            self.linear.weight = self.embedding.weight
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
        if self.multi_task:
            self.classifier = nn.Linear(hidden_size, 2)  # Dummy classification (even/odd)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        if self.embed_norm is not None:
            embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        # Apply LayerNorm and Dropout to the entire sequence output
        output = self.layer_norm(output)
        output = self.dropout(output)

        # Project to embedding dim if needed before decoding
        if self.decoder_proj is not None:
            decoded = self.decoder_proj(output)
        else:
            decoded = output
            
        logits = self.linear(decoded)
        
        if self.multi_task:
            # Use the final hidden state from the last layer for classification
            aux_logits = self.classifier(hidden[-1])
            return logits, aux_logits, hidden
        return logits, None, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
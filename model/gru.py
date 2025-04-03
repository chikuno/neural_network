import torch
import torch.nn as nn
import numpy as np

class GRUTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, multi_task=False):
        super(GRUTextGenerationModel, self).__init__()
        self.multi_task = multi_task
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        if self.multi_task:
            self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        logits = self.linear(output[:, -1, :])
        if self.multi_task:
            aux_logits = self.classifier(output[:, -1, :])
            return logits, aux_logits, hidden
        return logits, None, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(device)
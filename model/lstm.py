import torch
import torch.nn as nn
try:
    from config import config as cfg
except Exception:
    class _Dummy:
        pass
    cfg = _Dummy()
try:
    from model.embedding_block import ConvGLUBlock
except Exception:
    ConvGLUBlock = None

######################################
# LSTM-based Model
######################################
class LSTMTextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, multi_task=False, tie_weights=True):
        super(LSTMTextGenerationModel, self).__init__()
        self.multi_task = multi_task
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_norm = nn.LayerNorm(embedding_dim) if bool(getattr(cfg, 'EMBED_LAYER_NORM', False)) else None
        self.embed_dropout = nn.Dropout(dropout)
        # Optional embedding enhancement block
        use_block = bool(getattr(cfg, 'EMBED_CONV_BLOCK', False)) and ConvGLUBlock is not None
        self.embed_block = ConvGLUBlock(
            embedding_dim,
            channels=getattr(cfg, 'EMBED_CONV_CHANNELS', 512),
            kernel_size=getattr(cfg, 'EMBED_CONV_KERNEL', 3)
        ) if use_block else None
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Optional self-attention over LSTM outputs for richer long-range mixing
        self.use_output_attn = bool(getattr(cfg, 'USE_OUTPUT_ATTENTION', False))
        if self.use_output_attn:
            heads = int(getattr(cfg, 'OUTPUT_ATTENTION_HEADS', 4))
            # MultiheadAttention expects (S,B,E) by default; we'll transpose at use
            self.output_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=heads, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.tie_weights = tie_weights
        decoder_in = hidden_size
        self.decoder_proj = None
        if tie_weights:
            if hidden_size != embedding_dim:
                self.decoder_proj = nn.Linear(hidden_size, embedding_dim, bias=False)
                decoder_in = embedding_dim
            self.linear = nn.Linear(decoder_in, vocab_size, bias=False)
            self.linear.weight = self.embedding.weight
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
        if self.multi_task:
            self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        if self.embed_norm is not None:
            embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)
        if self.embed_block is not None:
            embedded = self.embed_block(embedded)
        output, hidden = self.lstm(embedded, hidden)
        # Apply optional self-attention across timesteps
        if self.use_output_attn:
            # output: (B, S, H) -> (S, B, H)
            attn_in = output.transpose(0, 1)
            attn_out, _ = self.output_attn(attn_in, attn_in, attn_in, need_weights=False)
            output = attn_out.transpose(0, 1)
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
            aux_logits = self.classifier(hidden[0][-1])
            return logits, aux_logits, hidden
        return logits, None, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))
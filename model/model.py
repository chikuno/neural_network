import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###############################
# Positional Encoding Module
###############################
class PositionalEncoding(nn.Module):
    """Implements the positional encoding as in 'Attention is All You Need'."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

###############################
# Advanced MLP
###############################
class AdvancedMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, 
                 activation='relu', batch_norm=True, initializer='he', 
                 multi_task=False, task_outputs=None, weight_decay=0.0, use_pos_enc=False):
        """
        Advanced MLP with configurable fully connected layers.
        Optionally, if use_pos_enc is True and the input is a sequence (3D tensor),
        positional encoding is applied and the sequence is flattened.
        """
        super(AdvancedMLP, self).__init__()
        self.use_pos_enc = use_pos_enc
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.initializer = initializer
        self.multi_task = multi_task
        self.task_outputs = task_outputs
        self.weight_decay = weight_decay

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(input_size, dropout=dropout)
        
        # Setup multi-task heads if needed.
        if self.multi_task:
            assert task_outputs is not None, "For multi-task learning, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in task_outputs])
        else:
            self.task_heads = [nn.Linear(hidden_layers[-1], output_size)]
        
        layers = []
        in_features = input_size
        # Note: if positional encoding is used, the forward pass will flatten the sequence,
        # so the first Linear layer expects input_dim = (seq_len * input_size). The user must
        # ensure consistency between training and inference.
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self.get_activation_function())
            layers.append(nn.Dropout(p=dropout))
            in_features = h
        if not self.multi_task:
            layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def get_activation_function(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            return nn.ReLU()

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # If input is a sequence and positional encoding is enabled, apply it.
        if self.use_pos_enc and x.dim() == 3:
            # x: (batch, seq_len, input_size) -> (seq_len, batch, input_size)
            x = x.transpose(0,1)
            x = self.pos_enc(x)
            x = x.transpose(0,1)
            # Flatten the sequence: (batch, seq_len * input_size)
            batch_size, seq_len, dim = x.size()
            x = x.reshape(batch_size, seq_len * dim)
        x = self.model(x)
        if self.multi_task:
            outputs = [head(x) for head in self.task_heads]
            return outputs
        else:
            return x
        
    def get_loss(self, output, target, criterion):
        if self.multi_task:
            losses = [criterion(task_output, target[i]) for i, task_output in enumerate(output)]
            return sum(losses)
        else:
            return criterion(output, target)

###############################
# Advanced GRU
###############################
class AdvancedGRU(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, 
                 initializer='he', multi_task=False, task_outputs=None, use_pos_enc=True):
        """
        Advanced GRU built from GRUCell layers.
        Allows each layer to have a different hidden size.
        If use_pos_enc is True and input is sequence (batch, seq_len, input_size), positional encoding is applied.
        """
        super(AdvancedGRU, self).__init__()
        self.use_pos_enc = use_pos_enc
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.dropout = dropout
        self.initializer = initializer
        self.multi_task = multi_task
        self.task_outputs = task_outputs

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(input_size, dropout=dropout)
        self.gru_cells = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_layers):
            in_dim = input_size if i == 0 else hidden_layers[i-1]
            self.gru_cells.append(nn.GRUCell(in_dim, hidden_size))
        if self.multi_task:
            assert task_outputs is not None, "For multi-task, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in task_outputs])
        else:
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.GRUCell):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if self.initializer == 'he':
                        nn.init.kaiming_normal_(param)
                    elif self.initializer == 'xavier':
                        nn.init.xavier_normal_(param)
                    elif self.initializer == 'normal':
                        nn.init.normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        if self.use_pos_enc and x.dim() == 3:
            x = x.transpose(0,1)
            x = self.pos_enc(x)
            x = x.transpose(0,1)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_layers]
        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.gru_cells):
                h[i] = cell(input_t, h[i])
                input_t = h[i]
                if i < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)
        if self.multi_task:
            outputs = [head(h[-1]) for head in self.task_heads]
            return outputs
        else:
            return self.output_layer(h[-1])
        
    def get_loss(self, output, target, criterion):
        if self.multi_task:
            losses = [criterion(task_output, target[i]) for i, task_output in enumerate(output)]
            return sum(losses)
        else:
            return criterion(output, target)

###############################
# Advanced LSTM
###############################
class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, 
                 initializer='he', multi_task=False, task_outputs=None, use_pos_enc=True):
        """
        Advanced LSTM built from LSTMCell layers.
        Allows each layer to have a different hidden size.
        If use_pos_enc is True and input is sequence, positional encoding is applied.
        """
        super(AdvancedLSTM, self).__init__()
        self.use_pos_enc = use_pos_enc
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.dropout = dropout
        self.initializer = initializer
        self.multi_task = multi_task
        self.task_outputs = task_outputs

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(input_size, dropout=dropout)
        self.lstm_cells = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_layers):
            in_dim = input_size if i == 0 else hidden_layers[i-1]
            self.lstm_cells.append(nn.LSTMCell(in_dim, hidden_size))
        if self.multi_task:
            assert task_outputs is not None, "For multi-task, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in task_outputs])
        else:
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.LSTMCell):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if self.initializer == 'he':
                        nn.init.kaiming_normal_(param)
                    elif self.initializer == 'xavier':
                        nn.init.xavier_normal_(param)
                    elif self.initializer == 'normal':
                        nn.init.normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        if self.use_pos_enc and x.dim() == 3:
            x = x.transpose(0,1)
            x = self.pos_enc(x)
            x = x.transpose(0,1)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_layers]
        c = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_layers]
        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]
                if i < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)
        if self.multi_task:
            outputs = [head(h[-1]) for head in self.task_heads]
            return outputs
        else:
            return self.output_layer(h[-1])
        
    def get_loss(self, output, target, criterion):
        if self.multi_task:
            losses = [criterion(task_output, target[i]) for i, task_output in enumerate(output)]
            return sum(losses)
        else:
            return criterion(output, target)

###############################
# Advanced RNN
###############################
class AdvancedRNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, 
                 activation='tanh', initializer='he', multi_task=False, task_outputs=None, use_pos_enc=True):
        """
        Advanced RNN built from RNNCell layers.
        Allows each layer to have a different hidden size.
        If use_pos_enc is True and input is sequence, positional encoding is applied.
        :param activation: Nonlinearity for the RNNCell (typically 'tanh' or 'relu').
        """
        super(AdvancedRNN, self).__init__()
        self.use_pos_enc = use_pos_enc
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.dropout = dropout
        self.activation = activation
        self.initializer = initializer
        self.multi_task = multi_task
        self.task_outputs = task_outputs

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(input_size, dropout=dropout)
        self.rnn_cells = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_layers):
            in_dim = input_size if i == 0 else hidden_layers[i-1]
            self.rnn_cells.append(nn.RNNCell(in_dim, hidden_size, nonlinearity=activation))
        if self.multi_task:
            assert task_outputs is not None, "For multi-task, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in task_outputs])
        else:
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.RNNCell):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if self.initializer == 'he':
                        nn.init.kaiming_normal_(param)
                    elif self.initializer == 'xavier':
                        nn.init.xavier_normal_(param)
                    elif self.initializer == 'normal':
                        nn.init.normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        if self.use_pos_enc and x.dim() == 3:
            x = x.transpose(0,1)
            x = self.pos_enc(x)
            x = x.transpose(0,1)
        batch_size, seq_len, _ = x.size()
        h = [torch.zeros(batch_size, hs, device=x.device) for hs in self.hidden_layers]
        for t in range(seq_len):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.rnn_cells):
                h[i] = cell(input_t, h[i])
                input_t = h[i]
                if i < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)
        if self.multi_task:
            outputs = [head(h[-1]) for head in self.task_heads]
            return outputs
        else:
            return self.output_layer(h[-1])
        
    def get_loss(self, output, target, criterion):
        if self.multi_task:
            losses = [criterion(task_output, target[i]) for i, task_output in enumerate(output)]
            return sum(losses)
        else:
            return criterion(output, target)

###############################
# Advanced Transformer
###############################
class AdvancedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, num_heads, dropout=0.5, 
                 initializer='he', multi_task=False, task_outputs=None, use_pos_enc=True):
        """
        Advanced Transformer encoder.
        :param input_dim: Dimension of input features.
        :param hidden_layers: List of hidden sizes for each encoder layer (first element is d_model).
        :param output_dim: Output dimension.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout rate.
        :param initializer: Weight initialization method.
        :param multi_task: Enable multi-task output.
        :param task_outputs: List of output sizes for multi-task.
        :param use_pos_enc: Whether to apply positional encoding after the embedding.
        """
        super(AdvancedTransformer, self).__init__()
        self.multi_task = multi_task
        self.task_outputs = task_outputs
        self.use_pos_enc = use_pos_enc
        # First element of hidden_layers defines d_model.
        d_model = hidden_layers[0]
        self.embedding = nn.Linear(input_dim, d_model)
        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.encoder_layers = nn.ModuleList()
        current_dim = d_model
        # Build encoder layers. If next layer hidden size differs, add a projection.
        for i, hidden_size in enumerate(hidden_layers):
            layer = nn.TransformerEncoderLayer(d_model=current_dim, nhead=num_heads, dropout=dropout)
            self.encoder_layers.append(layer)
            if i < len(hidden_layers) - 1 and hidden_layers[i+1] != current_dim:
                proj = nn.Linear(current_dim, hidden_layers[i+1])
                self.encoder_layers.append(proj)
                current_dim = hidden_layers[i+1]
        if self.multi_task:
            assert task_outputs is not None, "For multi-task, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(current_dim, out_size) for out_size in task_outputs])
        else:
            self.output_layer = nn.Linear(current_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.initializer = initializer
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        if self.use_pos_enc:
            x = x.transpose(0,1)  # (seq_len, batch, d_model)
            x = self.pos_enc(x)
            x = x.transpose(0,1)  # (batch, seq_len, d_model)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Pool over sequence length
        if self.multi_task:
            outputs = [head(x) for head in self.task_heads]
            return outputs
        else:
            return self.output_layer(x)
        
    def get_loss(self, output, target, criterion):
        if self.multi_task:
            losses = [criterion(task_output, target[i]) for i, task_output in enumerate(output)]
            return sum(losses)
        else:
            return criterion(output, target)

import torch
import torch.nn as nn

class AdvancedMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5, 
                 activation='relu', batch_norm=True, initializer='he', 
                 multi_task=False, task_outputs=None, weight_decay=0.0,
                 use_embedding=False, vocab_size=None, embedding_dim=None):
        """
        :param input_size: Number of input features
        :param hidden_layers: List of integers representing the number of neurons in each hidden layer
        :param output_size: Number of output features (for single-task or multi-task)
        :param dropout: Dropout rate (default 0.5)
        :param activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        :param batch_norm: Boolean for batch normalization after each layer
        :param initializer: Weight initialization method ('he', 'xavier', 'normal')
        :param multi_task: Boolean for enabling multi-task output
        :param task_outputs: List of output sizes for each task (for multi-task)
        :param weight_decay: L2 regularization strength (default 0.0)
        """
        super(AdvancedMLP, self).__init__()

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
        # Optional embedding configuration
        self.use_embedding = use_embedding
        if self.use_embedding:
            if vocab_size is None:
                raise ValueError('vocab_size must be provided if use_embedding is True')
            # default embedding dim to input_size if not provided
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim or input_size
            self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
            # adjust first linear layer input size to embedding_dim
            in_features = self.embedding_dim
        else:
            in_features = input_size

        # If multi-task, adjust output layer configurations
        if multi_task:
            assert task_outputs is not None, "For multi-task learning, task_outputs must be specified."
            self.task_heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in task_outputs])
        else:
            self.task_heads = [nn.Linear(hidden_layers[-1], output_size)]

        # Build the MLP network
        layers = []
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self.get_activation_function())
            layers.append(nn.Dropout(p=dropout))
            in_features = h
        
        # Output layer
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
        """
        Weight initialization
        """
        if isinstance(module, nn.Linear):
            if self.initializer == 'he':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif self.initializer == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif self.initializer == 'normal':
                nn.init.normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass of the MLP model
        """
        # If the model uses an embedding and x is an index tensor, embed first
        if self.use_embedding and x.dtype == torch.long:
            # x may be shape (1,) or (batch,); ensure long and move through embed
            emb = self.embed(x.to(torch.long))
            # embed returns (batch, embedding_dim) for indices; ensure correct shape
            x = emb

        x = self.model(x)
        
        if self.multi_task:
            # Output for each task head
            outputs = [head(x) for head in self.task_heads]
            return outputs
        else:
            return x
        
    def get_loss(self, output, target, criterion):
        """
        Compute loss for the MLP, supports multi-task learning.
        """
        if self.multi_task:
            losses = []
            for i, task_output in enumerate(output):
                loss = criterion(task_output, target[i])
                losses.append(loss)
            total_loss = sum(losses)
            return total_loss
        else:
            return criterion(output, target)

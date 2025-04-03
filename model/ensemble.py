# model/ensemble.py
import torch
import torch.nn as nn

class EnsembleVoting(nn.Module):
    """
    A meta-model that learns to weight the outputs from an ensemble of models.
    It takes a list of logits from individual models and outputs a final weighted prediction.
    """
    def __init__(self, num_models, vocab_size):
        super(EnsembleVoting, self).__init__()
        self.fc = nn.Linear(num_models, num_models)
        self.softmax = nn.Softmax(dim=1)
        self.vocab_size = vocab_size

    def forward(self, logits_list):
        """
        Args:
            logits_list: List of tensors, each of shape [batch_size, vocab_size].
        Returns:
            Weighted logits of shape [batch_size, vocab_size].
        """
        # Stack logits: [batch_size, num_models, vocab_size]
        stacked_logits = torch.stack(logits_list, dim=1)
        # Average across vocab dimension to get a score per model per sample
        model_scores = torch.mean(stacked_logits, dim=2)  # [batch_size, num_models]
        weights = self.softmax(self.fc(model_scores))      # [batch_size, num_models]
        # Weighted sum of logits
        weighted_logits = torch.sum(stacked_logits * weights.unsqueeze(2), dim=1)
        return weighted_logits

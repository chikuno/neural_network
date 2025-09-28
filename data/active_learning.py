# data/active_learning.py
import torch
import torch.nn.functional as F


def _ensure_batch(data_inputs: torch.Tensor) -> torch.Tensor:
    if data_inputs.dim() == 1:
        return data_inputs.unsqueeze(0)
    return data_inputs


def _sequence_reduce_logits(logits: torch.Tensor) -> torch.Tensor:
    """Reduce sequence logits to per-sample aggregate for uncertainty metrics.

    For (B,S,V) returns mean over S after computing per-token probabilities.
    """
    if logits.dim() == 3:
        return logits  # keep sequence shape; metrics handle
    return logits


def _entropy_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:  # (B,S,V)
        probs = F.softmax(logits, dim=-1)
        ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (B,S)
        return ent.mean(dim=1)  # (B,)
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)


def _margin_scores(logits: torch.Tensor) -> torch.Tensor:
    """Lower margin = more uncertain; we return negative margin so higher=more uncertain."""
    if logits.dim() == 3:
        # aggregate per-token margins then mean
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values  # (B,S,2)
        margins = top2[..., 0] - top2[..., 1]  # (B,S)
        per_sample = margins.mean(dim=1)  # (B,)
        return -per_sample
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values  # (B,2)
    margins = top2[:, 0] - top2[:, 1]
    return -margins


def _variation_ratio(logit_passes: torch.Tensor) -> torch.Tensor:
    """Variation ratio from multiple stochastic passes.

    logit_passes: (T,B,V) or (T,B,S,V). We take argmax per pass, per sample (and optionally per token),
    then compute 1 - (mode_frequency / T). Sequence case averages VR over tokens.
    """
    if logit_passes.dim() == 4:  # (T,B,S,V)
        preds = logit_passes.argmax(dim=-1)  # (T,B,S)
        T = preds.size(0)
        # mode frequency per token
        mode_freq = []
        for t in range(preds.size(2)):
            token_preds = preds[:, :, t]  # (T,B)
            # count mode per sample
            vals, _ = torch.mode(token_preds, dim=0)
            # frequency of mode per sample
            freq = torch.stack([(token_preds[:, i] == vals[i]).sum() for i in range(token_preds.size(1))])
            mode_freq.append(freq)
        mode_freq = torch.stack(mode_freq, dim=1).float()  # (B,S)
        vr = 1.0 - (mode_freq / T)
        return vr.mean(dim=1)
    elif logit_passes.dim() == 3:  # (T,B,V)
        preds = logit_passes.argmax(dim=-1)  # (T,B)
        T = preds.size(0)
        vals, _ = torch.mode(preds, dim=0)
        freq = torch.stack([(preds[:, i] == vals[i]).sum() for i in range(preds.size(1))]).float()
        return 1.0 - (freq / T)
    else:
        raise ValueError("Unexpected shape for logit_passes")


def _bald_scores(logit_passes: torch.Tensor) -> torch.Tensor:
    """Compute BALD = H[mean_p] - mean_t H[p_t].

    logit_passes: (T,B,V) or (T,B,S,V). Sequence variant averages token BALD.
    Returns: (B,)
    """
    if logit_passes.dim() == 4:  # (T,B,S,V)
        probs = torch.softmax(logit_passes, dim=-1)  # (T,B,S,V)
        mean_p = probs.mean(dim=0)  # (B,S,V)
        # H[mean_p]
        entropy_mean = -torch.sum(mean_p * torch.log(mean_p + 1e-9), dim=-1)  # (B,S)
        # mean_t H[p_t]
        ent_t = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (T,B,S)
        expected_ent = ent_t.mean(dim=0)  # (B,S)
        bald = entropy_mean - expected_ent
        return bald.mean(dim=1)
    elif logit_passes.dim() == 3:  # (T,B,V)
        probs = torch.softmax(logit_passes, dim=-1)  # (T,B,V)
        mean_p = probs.mean(dim=0)  # (B,V)
        entropy_mean = -torch.sum(mean_p * torch.log(mean_p + 1e-9), dim=-1)  # (B,)
        ent_t = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (T,B)
        expected_ent = ent_t.mean(dim=0)  # (B,)
        bald = entropy_mean - expected_ent
        return bald
    else:
        raise ValueError("Unexpected shape for logit_passes in BALD")


def _forward_logits(model, data_inputs: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        try:
            outputs = model(data_inputs.to(device))
        except TypeError:
            hidden = model.init_hidden(data_inputs.size(0), device)
            outputs = model(data_inputs.to(device), hidden)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    return logits


def select_uncertain_samples(model, data_inputs, top_k=10, device='cpu', return_indices: bool=False,
                             strategy: str = 'entropy', mc_passes: int = 5, enable_mc_dropout: bool = True):
    """Generalized uncertainty sampling.

    strategy: 'entropy' | 'margin' | 'variation_ratio' | 'bald'
    mc_passes: number of stochastic forward passes for variation ratio.
    enable_mc_dropout: if True, sets model.train() for MC passes (restored after).
    """
    data_inputs = _ensure_batch(data_inputs)
    if data_inputs is None or data_inputs.numel() == 0:
        return (data_inputs, torch.empty(0, dtype=torch.long)) if return_indices else data_inputs

    batch_size = data_inputs.size(0)
    k = min(int(top_k), batch_size) if top_k is not None else batch_size
    if k <= 0:
        return (data_inputs[:0], torch.empty(0, dtype=torch.long)) if return_indices else data_inputs[:0]

    model_was_training = model.training

    if strategy in ('variation_ratio', 'bald'):
        # MC Dropout passes
        if enable_mc_dropout:
            model.train()
        passes = []
        for _ in range(mc_passes):
            logits = _forward_logits(model, data_inputs, device)
            passes.append(logits.unsqueeze(0))
        logit_passes = torch.cat(passes, dim=0)
        if strategy == 'variation_ratio':
            scores = _variation_ratio(logit_passes)
        else:
            scores = _bald_scores(logit_passes)
    else:
        model.eval()
        logits = _forward_logits(model, data_inputs, device)
        if strategy == 'entropy':
            scores = _entropy_scores(logits)
        elif strategy == 'margin':
            scores = _margin_scores(logits)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # Restore original training mode
    if model_was_training:
        model.train()
    else:
        model.eval()

    if scores.dim() != 1 or scores.size(0) != batch_size:
        scores = scores.view(batch_size, -1).mean(dim=1)

    sorted_indices = torch.argsort(scores, descending=True)
    sel_idx = sorted_indices[:k]
    sel_idx = sel_idx[sel_idx < batch_size]
    selected = data_inputs[sel_idx]
    return (selected, sel_idx) if return_indices else selected

# Backwards compatibility alias
def select_uncertain_samples_entropy(model, data_inputs, top_k=10, device='cpu'):
    return select_uncertain_samples(model, data_inputs, top_k=10, device=device, strategy='entropy')

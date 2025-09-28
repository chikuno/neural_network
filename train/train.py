import torch
from tqdm import tqdm

def evaluate(model, data_loader, criterion, aux_criterion, device, model_type='lstm', multi_task=False):
    model.eval()
    total_loss = 0
    total_aux_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = len(data_loader)
    if num_batches == 0:
        return 0.0, None, 0.0

    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            # Prepare inputs for the model: only convert to float for models
            # that do NOT expose an embedding layer. Transformers/RNNs usually
            # provide an embedding (named 'embedding' or 'embed'), so leave
            # their inputs as integer token indices (LongTensor).
            # Since all models now use embeddings, the complex one-hot encoding logic
            # for MLPs is no longer needed. All models will receive LongTensors.
            if not (hasattr(model, 'embedding') or hasattr(model, 'embed')):
                 batch_inputs = batch_inputs.float()

            # Normalize model outputs: support multiple return signatures
            if hasattr(model, 'init_hidden'):
                hidden = model.init_hidden(batch_inputs.size(0), device)
                out = model(batch_inputs, hidden)
            else:
                # For transformer, provide padding mask so PAD tokens are ignored
                if model_type == 'transformer':
                    try:
                        from config import config as cfg
                        pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                    except Exception:
                        pad_idx = 0
                    src_key_padding_mask = (batch_inputs == pad_idx)
                    out = model(batch_inputs, src_key_padding_mask=src_key_padding_mask)
                else:
                    out = model(batch_inputs)

            # Normalize out into (logits, aux_logits)
            logits = None
            aux_logits = None
            if isinstance(out, tuple) or isinstance(out, list):
                if len(out) == 1:
                    logits = out[0]
                elif len(out) == 2:
                    logits, aux_logits = out
                elif len(out) == 3:
                    logits, aux_logits, _ = out
                else:
                    logits = out[0]
                    aux_logits = None
            else:
                logits = out

            # If logits are sequence-level (B, S, V) and targets are (B, S), compute token-wise loss
            if logits is not None and logits.dim() == 3 and batch_targets.dim() == 2:
                # Align sequence lengths just in case
                S = min(logits.size(1), batch_targets.size(1))
                if logits.size(1) != S:
                    logits = logits[:, :S, :]
                if batch_targets.size(1) != S:
                    batch_targets = batch_targets[:, :S]
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
            else:
                # Fallback: (B,V) vs (B,) path (e.g., MLP)
                logits_for_loss = logits[:, -1, :] if (logits is not None and logits.dim() == 3) else logits
                loss = criterion(logits_for_loss, batch_targets if batch_targets.dim()==1 else batch_targets[:, -1])
            total_loss += loss.item()

            # Calculate accuracy
            if logits is not None:
                # Reshape logits and targets for accuracy calculation if needed
                if logits.dim() == 3 and batch_targets.dim() == 2:  # (B,S,V) vs (B,S)
                    S = min(logits.size(1), batch_targets.size(1))
                    preds = torch.argmax(logits[:, :S, :], dim=-1)
                    target_slice = batch_targets[:, :S]
                    total_correct += (preds == target_slice).sum().item()
                    total_tokens += target_slice.numel()
                elif logits.dim() == 2 and batch_targets.dim() == 1:  # (B,V) vs (B)
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == batch_targets).sum().item()
                    total_tokens += batch_targets.numel()
                else:
                    # Last-step compare
                    last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
                    last_targets = batch_targets[:, -1] if batch_targets.dim() == 2 else batch_targets
                    preds = torch.argmax(last_logits, dim=1)
                    total_correct += (preds == last_targets).sum().item()
                    total_tokens += last_targets.numel()

            if multi_task and aux_logits is not None:
                # Use last-step labels for aux task when targets are (B,S)
                aux_base = batch_targets[:, -1] if batch_targets.dim() == 2 else batch_targets
                aux_labels = (aux_base % 2).to(device)
                aux_loss = aux_criterion(aux_logits, aux_labels)
                total_aux_loss += aux_loss.item()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_aux_loss = (total_aux_loss / num_batches) if (multi_task and num_batches > 0) else None
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_aux_loss, accuracy

def train(model, train_loader, val_loader,
          optimizer, criterion, aux_criterion, epochs, batch_size, device,
          clip=5, early_stop_patience=3, checkpoint_path='saved_models/best_model.pth',
          model_type='lstm', multi_task=False, writer=None, scheduler=None, scaler=None,
          max_batches=None, grad_accum_steps=1, holdout_loader=None):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_aux_loss = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        # Curriculum: progressively increase effective sequence length
        eff_len = None
        try:
            from config import config as cfg
            if getattr(cfg, 'CURRICULUM', False):
                min_len = int(getattr(cfg, 'MIN_SEQUENCE_LENGTH', 16))
                full_len = int(getattr(cfg, 'SEQUENCE_LENGTH', 30))
                grow_epochs = max(1, int(getattr(cfg, 'CURRICULUM_EPOCHS', 3)))
                ratio = min(1.0, (epoch + 1) / float(grow_epochs))
                eff_len = max(2, int(min_len + ratio * (full_len - min_len)))
        except Exception:
            eff_len = None
        
        num_batches = len(train_loader)
        if num_batches == 0:
            print("Warning: training data loader is empty. Skipping training epoch.")
            return 0.0

        effective_batches = min(num_batches, max_batches) if max_batches else num_batches
        progress_bar = tqdm(enumerate(train_loader), total=effective_batches, desc=f"Epoch {epoch+1}")
        # Optional label smoothing anneal
        try:
            from config import config as cfg
            if getattr(cfg, 'LABEL_SMOOTHING_ANNEAL', False):
                # linearly anneal from START to END across epochs
                start = float(getattr(cfg, 'LABEL_SMOOTHING_START', 0.05))
                end = float(getattr(cfg, 'LABEL_SMOOTHING_END', 0.0))
                t = epoch / max(1, epochs-1)
                cur_smooth = start + (end - start) * t
                if hasattr(criterion, 'label_smoothing'):
                    criterion.label_smoothing = max(0.0, float(cur_smooth))
        except Exception:
            pass

        accum_count = 0
        optimizer.zero_grad()
        for batch_idx, (batch_inputs, batch_targets) in progress_bar:
            if max_batches and batch_idx >= max_batches:
                break
            
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            # Apply effective length cropping under curriculum
            if eff_len is not None and batch_inputs.dim() == 2 and batch_targets.dim() in (1,2):
                if batch_inputs.size(1) > eff_len:
                    batch_inputs = batch_inputs[:, :eff_len]
                if batch_targets.dim() == 2 and batch_targets.size(1) > eff_len:
                    batch_targets = batch_targets[:, :eff_len]
                elif batch_targets.dim() == 1 and batch_inputs.size(1) > 0:
                    # Last-token target path remains valid; no change needed
                    pass

            # Mixed precision training (if scaler is provided)
            if scaler:
                with torch.cuda.amp.autocast():
                    if not (hasattr(model, 'embedding') or hasattr(model, 'embed')):
                        batch_inputs = batch_inputs.float()

                    if hasattr(model, 'init_hidden'):
                        hidden = model.init_hidden(batch_inputs.size(0), device)
                        out = model(batch_inputs, hidden)
                    else:
                        out = model(batch_inputs)

                    logits, aux_logits = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
                    # Full-sequence loss if possible
                    if logits is not None and logits.dim() == 3 and batch_targets.dim() == 2:
                        S = min(logits.size(1), batch_targets.size(1))
                        loss = criterion(logits[:, :S, :].reshape(-1, logits.size(-1)), batch_targets[:, :S].reshape(-1))
                    else:
                        logits_for_loss = logits[:, -1, :] if (logits is not None and logits.dim() == 3) else logits
                        target_last = batch_targets if batch_targets.dim()==1 else batch_targets[:, -1]
                        loss = criterion(logits_for_loss, target_last)
                    # Include auxiliary loss if enabled
                    total_loss = loss
                    if multi_task and aux_logits is not None:
                        aux_base = batch_targets[:, -1] if batch_targets.dim() == 2 else batch_targets
                        aux_labels = (aux_base % 2).to(device)
                        aux_loss = aux_criterion(aux_logits, aux_labels)
                        total_loss = loss + 0.5 * aux_loss
                        epoch_aux_loss += aux_loss.item()
                    # Temporal regularization on consecutive logits (smoothness)
                    try:
                        from config import config as cfg
                        lam = float(getattr(cfg, 'LOSS_TAR_LAMBDA', 0.0))
                        if lam > 0 and logits is not None and logits.dim() == 3 and logits.size(1) > 1:
                            # L2 on logits diffs over time (batch, seq-1, vocab)
                            diffs = (logits[:, 1:, :] - logits[:, :-1, :])
                            tar = lam * (diffs.pow(2).mean())
                            total_loss = total_loss + tar
                    except Exception:
                        pass
                    # Normalize by accumulation steps
                    total_loss = total_loss / max(1, grad_accum_steps)
                scaler.scale(total_loss).backward()
                accum_count += 1
                if accum_count % max(1, grad_accum_steps) == 0:
                    # Unscale then clip before stepping
                    try:
                        scaler.unscale_(optimizer)
                        # Compute total grad norm after unscale for logging
                        try:
                            total_norm = torch.norm(torch.stack([
                                p.grad.norm(2) for p in model.parameters() if p.grad is not None
                            ]), 2).item()
                            grad_norm_sum += float(total_norm)
                            grad_norm_count += 1
                            try:
                                from config import config as cfg
                                warn_thr = float(getattr(cfg, 'GRAD_NORM_WARN', 0.0) or 0.0)
                                if warn_thr > 0 and total_norm > warn_thr:
                                    print(f"Warning: large grad norm {total_norm:.2f} > {warn_thr}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # NaN/Inf guard
                        try:
                            from config import config as cfg
                            if getattr(cfg, 'GRAD_NAN_PROTECT', False):
                                for p in model.parameters():
                                    if p.grad is not None and torch.isnan(p.grad).any():
                                        p.grad[torch.isnan(p.grad)] = 0.0
                                    if p.grad is not None and torch.isinf(p.grad).any():
                                        p.grad[torch.isinf(p.grad)] = 0.0
                        except Exception:
                            pass
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    except Exception:
                        pass
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                if not (hasattr(model, 'embedding') or hasattr(model, 'embed')):
                    batch_inputs = batch_inputs.float()

                if hasattr(model, 'init_hidden'):
                    hidden = model.init_hidden(batch_inputs.size(0), device)
                    out = model(batch_inputs, hidden)
                else:
                    if model_type == 'transformer':
                        try:
                            from config import config as cfg
                            pad_idx = int(getattr(cfg, 'PAD_IDX', 0))
                        except Exception:
                            pad_idx = 0
                        src_key_padding_mask = (batch_inputs == pad_idx)
                        out = model(batch_inputs, src_key_padding_mask=src_key_padding_mask)
                    else:
                        out = model(batch_inputs)

                logits, aux_logits = (out[0], out[1]) if isinstance(out, tuple) else (out, None)

                if logits is not None and logits.dim() == 3 and batch_targets.dim() == 2:
                    S = min(logits.size(1), batch_targets.size(1))
                    loss = criterion(logits[:, :S, :].reshape(-1, logits.size(-1)), batch_targets[:, :S].reshape(-1))
                else:
                    logits_for_loss = logits[:, -1, :] if (logits is not None and logits.dim() == 3) else logits
                    target_last = batch_targets if batch_targets.dim()==1 else batch_targets[:, -1]
                    loss = criterion(logits_for_loss, target_last)
                total_loss = loss
                if multi_task and aux_logits is not None:
                    aux_base = batch_targets[:, -1] if batch_targets.dim() == 2 else batch_targets
                    aux_labels = (aux_base % 2).to(device)
                    aux_loss = aux_criterion(aux_logits, aux_labels)
                    total_loss = loss + 0.5 * aux_loss
                    epoch_aux_loss += aux_loss.item()
                # Temporal regularization on consecutive logits (smoothness)
                try:
                    from config import config as cfg
                    lam = float(getattr(cfg, 'LOSS_TAR_LAMBDA', 0.0))
                    if lam > 0 and logits is not None and logits.dim() == 3 and logits.size(1) > 1:
                        diffs = (logits[:, 1:, :] - logits[:, :-1, :])
                        tar = lam * (diffs.pow(2).mean())
                        total_loss = total_loss + tar
                except Exception:
                    pass
                # Normalize by accumulation steps
                total_loss = total_loss / max(1, grad_accum_steps)
                total_loss.backward()
                accum_count += 1
                if accum_count % max(1, grad_accum_steps) == 0:
                    # Compute grad norm for logging
                    try:
                        total_norm = torch.norm(torch.stack([
                            p.grad.norm(2) for p in model.parameters() if p.grad is not None
                        ]), 2).item()
                        grad_norm_sum += float(total_norm)
                        grad_norm_count += 1
                        try:
                            from config import config as cfg
                            warn_thr = float(getattr(cfg, 'GRAD_NORM_WARN', 0.0) or 0.0)
                            if warn_thr > 0 and total_norm > warn_thr:
                                print(f"Warning: large grad norm {total_norm:.2f} > {warn_thr}")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # NaN/Inf guard
                    try:
                        from config import config as cfg
                        if getattr(cfg, 'GRAD_NAN_PROTECT', False):
                            for p in model.parameters():
                                if p.grad is not None and torch.isnan(p.grad).any():
                                    p.grad[torch.isnan(p.grad)] = 0.0
                                if p.grad is not None and torch.isinf(p.grad).any():
                                    p.grad[torch.isinf(p.grad)] = 0.0
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))

        # Flush any remaining gradients if the epoch ended mid-accumulation
        if accum_count % max(1, grad_accum_steps) != 0:
            try:
                if scaler:
                    try:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    except Exception:
                        pass
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
            finally:
                optimizer.zero_grad()

        # Evaluate model on validation (optionally with EMA weights)
        used_ema = False
        try:
            # If optimizer exposes EMA and configured for eval-only, swap
            if hasattr(optimizer, 'use_ema') and optimizer.use_ema and getattr(optimizer, 'ema_eval_only', True):
                optimizer.swap_to_ema()
                used_ema = True
        except Exception:
            used_ema = False

        val_loss, val_aux_loss, val_acc = evaluate(model, val_loader, criterion, aux_criterion, device, model_type, multi_task)
        # Optional holdout eval also under EMA weights
        holdout_snapshot_loss = None
        if holdout_loader is not None:
            try:
                holdout_snapshot_loss, _, _ = evaluate(model, holdout_loader, criterion, aux_criterion, device, model_type, multi_task)
            except Exception:
                holdout_snapshot_loss = None

        # Restore original (non-EMA) weights after evaluation if we swapped
        if used_ema:
            try:
                optimizer.restore_from_ema()
            except Exception:
                pass
        train_loss = epoch_loss/num_batches
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        val_ppl = torch.exp(torch.tensor(val_loss)).item() if val_loss > 0 else float('inf')
        holdout_ppl = None
        holdout_loss = None
        use_holdout = False
        try:
            from config import config as cfg
            use_holdout = getattr(cfg, 'EARLY_STOP_METRIC', 'val_loss') == 'holdout_ppl'
        except Exception:
            use_holdout = False
        if holdout_loader is not None and use_holdout:
            # Reuse snapshot loss if already computed under EMA; otherwise compute
            if holdout_snapshot_loss is not None:
                h_loss = holdout_snapshot_loss
            else:
                h_loss, _, _ = evaluate(model, holdout_loader, criterion, aux_criterion, device, model_type, multi_task)
            holdout_loss = h_loss
            holdout_ppl = torch.exp(torch.tensor(h_loss)).item() if h_loss > 0 else float('inf')
        # Now that we have val_loss, step schedulers appropriately
        try:
            if scheduler is not None:
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step(train_loss)
            elif hasattr(optimizer, 'scheduler_type') and optimizer.scheduler_type == 'plateau':
                optimizer.scheduler.step(val_loss)
        except Exception:
            pass

        print(f"\nEpoch {epoch+1} Training Loss: {train_loss:.4f} (PPL {train_ppl:.2f}) | Validation Loss: {val_loss:.4f} (PPL {val_ppl:.2f}) | Validation Acc: {val_acc:.4f}")
        if holdout_ppl is not None:
            print(f"           Holdout Loss: {holdout_loss:.4f} (PPL {holdout_ppl:.2f})")
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Loss/Validation', val_loss, epoch+1)
            writer.add_scalar('PPL/Train', train_ppl, epoch+1)
            writer.add_scalar('PPL/Validation', val_ppl, epoch+1)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch+1)
            if holdout_ppl is not None:
                writer.add_scalar('Loss/Holdout', float(holdout_loss), epoch+1)
                writer.add_scalar('PPL/Holdout', float(holdout_ppl), epoch+1)
            # Log average gradient norm for the epoch (if any measurements captured)
            try:
                if grad_norm_count > 0:
                    writer.add_scalar('GradNorm/Avg', grad_norm_sum / float(max(1, grad_norm_count)), epoch+1)
            except Exception:
                pass
            # Log parameter histograms (weights only) once per epoch
            try:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param is not None and param.data is not None:
                            writer.add_histogram(f'Params/{name}', param.data, epoch+1)
            except Exception:
                pass

        # Save last checkpoint for potential resume
        try:
            # Derive a stable 'last' filename alongside 'best'
            if 'best_model_' in checkpoint_path:
                name = checkpoint_path.split('best_model_')[-1].split('.pth')[0]
                last_path = f"saved_models/last_model_{name}.pth"
            else:
                last_path = checkpoint_path + '.last'
            # Try to embed vocabulary mappings for robust inference restores
            try:
                from data.data import try_load_vocab
                w2i, i2w = try_load_vocab()
            except Exception:
                w2i, i2w = None, None
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'word_to_index': w2i,
                'index_to_word': i2w,
            }
            torch.save(state, last_path)
        except Exception:
            pass

        metric_for_early = val_loss
        if holdout_loss is not None:
            metric_for_early = holdout_loss
        if metric_for_early < best_val_loss:
            best_val_loss = metric_for_early
            try:
                # Try to embed vocabulary mappings for robust inference restores
                try:
                    from data.data import try_load_vocab
                    w2i, i2w = try_load_vocab()
                except Exception:
                    w2i, i2w = None, None
                state_best = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'word_to_index': w2i,
                    'index_to_word': i2w,
                }
                torch.save(state_best, checkpoint_path)
            except Exception:
                torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with monitored loss {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return epoch_loss/num_batches

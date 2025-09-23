import torch
from tqdm import tqdm

def evaluate(model, inputs, targets, criterion, aux_criterion, batch_size, device, model_type='lstm', multi_task=False):
    model.eval()
    total_loss = 0
    total_aux_loss = 0
    dataset_len = len(inputs)
    if dataset_len == 0:
        return 0.0, None
    num_batches = (dataset_len + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, dataset_len)
            batch_inputs, batch_targets = inputs[start:end], targets[start:end]
            # Prepare inputs for the model: only convert to float for models
            # that do NOT expose an embedding layer. Transformers/RNNs usually
            # provide an embedding (named 'embedding' or 'embed'), so leave
            # their inputs as integer token indices (LongTensor).
            has_embedding = hasattr(model, 'embedding') or hasattr(model, 'embed') or getattr(model, 'use_embedding', False)
            if not hasattr(model, 'init_hidden') and not has_embedding:
                # If the model specifies an input_size (MLP expecting flattened one-hot
                # vectors across the sequence), convert token index tensor of shape
                # (batch, seq_len) into a float one-hot flattened tensor of shape
                # (batch, input_size). Otherwise fall back to casting to float.
                if hasattr(model, 'input_size') and batch_inputs.dim() == 2 and batch_inputs.dtype in (torch.long, torch.int64):
                    seq_len = batch_inputs.size(1)
                    input_size = getattr(model, 'input_size')
                    # Determine token vocab from data and desired token_vocab to match input_size
                    batch_inputs = batch_inputs.to(torch.long)
                    data_vocab = int(batch_inputs.max().item()) + 1
                    desired_vocab = (input_size + seq_len - 1) // seq_len  # ceil div
                    token_vocab = max(data_vocab, desired_vocab)
                    one_hot = torch.zeros(batch_inputs.size(0), seq_len, token_vocab, device=device, dtype=torch.float)
                    one_hot.scatter_(2, batch_inputs.unsqueeze(2), 1.0)
                    flat = one_hot.view(batch_inputs.size(0), -1)
                    # Pad or trim to match model.input_size
                    if flat.size(1) > input_size:
                        batch_inputs = flat[:, :input_size]
                    elif flat.size(1) < input_size:
                        pad = torch.zeros(batch_inputs.size(0), input_size - flat.size(1), device=device, dtype=torch.float)
                        batch_inputs = torch.cat([flat, pad], dim=1)
                    else:
                        batch_inputs = flat
                else:
                    # convert index/long tensors to float (e.g., one-hot or float inputs expected by other MLPs)
                    batch_inputs = batch_inputs.float()

            # Normalize model outputs: support multiple return signatures
            if hasattr(model, 'init_hidden'):
                hidden = model.init_hidden(batch_inputs.size(0), device)
                out = model(batch_inputs, hidden)
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
                    # List of task heads (multi-task) -> concatenate or take first as main logits
                    logits = out[0]
                    aux_logits = None
            else:
                logits = out
            loss = criterion(logits, batch_targets)
            total_loss += loss.item()
            if multi_task and aux_logits is not None:
                aux_labels = (batch_targets % 2).to(device)
                aux_loss = aux_criterion(aux_logits, aux_labels)
                total_aux_loss += aux_loss.item()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_aux_loss = (total_aux_loss / num_batches) if (multi_task and num_batches > 0) else None
    return avg_loss, avg_aux_loss

def train(model, train_inputs, train_targets, val_inputs, val_targets,
          optimizer, criterion, aux_criterion, epochs, batch_size, device,
          clip=5, early_stop_patience=3, checkpoint_path='saved_models/best_model.pth',
          model_type='lstm', multi_task=False, writer=None, scheduler=None, scaler=None):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_aux_loss = 0
        dataset_size = len(train_inputs)
        if dataset_size == 0:
            print("Warning: training dataset is empty. Skipping training epoch.")
            return 0.0

        indices = torch.randperm(dataset_size)
        train_inputs, train_targets = train_inputs[indices], train_targets[indices]
        num_batches = (dataset_size + batch_size - 1) // batch_size

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            start = batch * batch_size
            end = min(start + batch_size, dataset_size)
            batch_inputs = train_inputs[start:end]
            batch_targets = train_targets[start:end]

            optimizer.zero_grad()  # Note: MetaOptimizer holds the underlying optimizer
            # Mixed precision training (if scaler is provided)
            if scaler:
                with torch.cuda.amp.autocast():
                    has_embedding = hasattr(model, 'embedding') or hasattr(model, 'embed') or getattr(model, 'use_embedding', False)
                    if not hasattr(model, 'init_hidden') and not has_embedding:
                        if hasattr(model, 'input_size') and batch_inputs.dim() == 2 and batch_inputs.dtype in (torch.long, torch.int64):
                            seq_len = batch_inputs.size(1)
                            input_size = getattr(model, 'input_size')
                            batch_inputs = batch_inputs.to(torch.long)
                            data_vocab = int(batch_inputs.max().item()) + 1
                            desired_vocab = (input_size + seq_len - 1) // seq_len
                            token_vocab = max(data_vocab, desired_vocab)
                            one_hot = torch.zeros(batch_inputs.size(0), seq_len, token_vocab, device=device, dtype=torch.float)
                            one_hot.scatter_(2, batch_inputs.unsqueeze(2), 1.0)
                            flat = one_hot.view(batch_inputs.size(0), -1)
                            if flat.size(1) > input_size:
                                batch_inputs = flat[:, :input_size]
                            elif flat.size(1) < input_size:
                                pad = torch.zeros(batch_inputs.size(0), input_size - flat.size(1), device=device, dtype=torch.float)
                                batch_inputs = torch.cat([flat, pad], dim=1)
                            else:
                                batch_inputs = flat
                        else:
                            batch_inputs = batch_inputs.float()

                    if hasattr(model, 'init_hidden'):
                        hidden = model.init_hidden(batch_inputs.size(0), device)
                        out = model(batch_inputs, hidden)
                    else:
                        out = model(batch_inputs)

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
                    loss = criterion(logits, batch_targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                has_embedding = hasattr(model, 'embedding') or hasattr(model, 'embed') or getattr(model, 'use_embedding', False)
                if not hasattr(model, 'init_hidden') and not has_embedding:
                    if hasattr(model, 'input_size') and batch_inputs.dim() == 2 and batch_inputs.dtype in (torch.long, torch.int64):
                        seq_len = batch_inputs.size(1)
                        input_size = getattr(model, 'input_size')
                        batch_inputs = batch_inputs.to(torch.long)
                        data_vocab = int(batch_inputs.max().item()) + 1
                        desired_vocab = (input_size + seq_len - 1) // seq_len
                        token_vocab = max(data_vocab, desired_vocab)
                        one_hot = torch.zeros(batch_inputs.size(0), seq_len, token_vocab, device=device, dtype=torch.float)
                        one_hot.scatter_(2, batch_inputs.unsqueeze(2), 1.0)
                        flat = one_hot.view(batch_inputs.size(0), -1)
                        if flat.size(1) > input_size:
                            batch_inputs = flat[:, :input_size]
                        elif flat.size(1) < input_size:
                            pad = torch.zeros(batch_inputs.size(0), input_size - flat.size(1), device=device, dtype=torch.float)
                            batch_inputs = torch.cat([flat, pad], dim=1)
                        else:
                            batch_inputs = flat
                    else:
                        batch_inputs = batch_inputs.float()

                if hasattr(model, 'init_hidden'):
                    hidden = model.init_hidden(batch_inputs.size(0), device)
                    out = model(batch_inputs, hidden)
                else:
                    out = model(batch_inputs)

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
                loss = criterion(logits, batch_targets)
                total_loss = loss
                if multi_task and aux_logits is not None:
                    aux_labels = (batch_targets % 2).to(device)
                    aux_loss = aux_criterion(aux_logits, aux_labels)
                    total_loss = loss + 0.5 * aux_loss
                    epoch_aux_loss += aux_loss.item()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss/(batch+1))

        if scheduler:
            scheduler.step(epoch_loss/num_batches)

        # Evaluate model on validation set
        val_loss, val_aux_loss = evaluate(model, val_inputs, val_targets, criterion, aux_criterion, batch_size, device, model_type, multi_task)
        print(f"\nEpoch {epoch+1} Training Loss: {epoch_loss/num_batches:.4f} | Validation Loss: {val_loss:.4f}")
        if writer:
            writer.add_scalar('Loss/Train', epoch_loss/num_batches, epoch+1)
            writer.add_scalar('Loss/Validation', val_loss, epoch+1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with validation loss {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return epoch_loss/num_batches

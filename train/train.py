import torch
from tqdm import tqdm

def evaluate(model, inputs, targets, criterion, aux_criterion, batch_size, device, model_type='lstm', multi_task=False):
    model.eval()
    total_loss = 0
    total_aux_loss = 0
    num_batches = len(inputs) // batch_size
    with torch.no_grad():
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_inputs, batch_targets = inputs[start:end], targets[start:end]
            if model_type in ['rnn', 'gru', 'lstm', 'mlp']:
                hidden = model.init_hidden(batch_inputs.size(0), device)
                logits, aux_logits, _ = model(batch_inputs, hidden)
            else:
                logits, aux_logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            total_loss += loss.item()
            if multi_task and aux_logits is not None:
                aux_labels = (batch_targets % 2).to(device)
                aux_loss = aux_criterion(aux_logits, aux_labels)
                total_aux_loss += aux_loss.item()
    avg_loss = total_loss / num_batches
    avg_aux_loss = (total_aux_loss / num_batches) if multi_task else None
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
        indices = torch.randperm(dataset_size)
        train_inputs, train_targets = train_inputs[indices], train_targets[indices]
        num_batches = dataset_size // batch_size

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_inputs = train_inputs[start:end]
            batch_targets = train_targets[start:end]

            optimizer.zero_grad()  # Note: MetaOptimizer holds the underlying optimizer
            # Mixed precision training (if scaler is provided)
            if scaler:
                with torch.cuda.amp.autocast():
                    if model_type in ['rnn', 'gru', 'lstm', 'mlp']:
                        hidden = model.init_hidden(batch_inputs.size(0), device)
                        logits, aux_logits, _ = model(batch_inputs, hidden)
                    else:
                        logits, aux_logits = model(batch_inputs)
                    loss = criterion(logits, batch_targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if model_type in ['rnn', 'gru', 'lstm', 'mlp']:
                    hidden = model.init_hidden(batch_inputs.size(0), device)
                    logits, aux_logits, _ = model(batch_inputs, hidden)
                else:
                    logits, aux_logits = model(batch_inputs)
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

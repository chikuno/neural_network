
#Module imports
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json

# Import custom modules
from config import config
import data.data as data
from train.train import train as train_run
import model
# 'train' and 'inference' modules imported implicitly via specific imports above
from inference.generate import generate_text_simple
import utils
import data.augmentation as augmentation
import data.active_learning as active_learning
from model.pid_controller import PIDController
from train.optimizer import MetaOptimizer


def safe_load_checkpoint(model, checkpoint_path):
    """Attempt to load checkpoint into model but only copy parameters that match in name and shape.
    Prints a summary of loaded/skipped keys and returns True if any keys were loaded.
    """
    if not os.path.exists(checkpoint_path):
        return False
    try:
        ckpt = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to read checkpoint {checkpoint_path}: {e}")
        return False

    model_state = model.state_dict()
    loaded = False
    for k, v in ckpt.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                loaded = True
            else:
                print(f"Skipping param {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
        else:
            print(f"Skipping param {k}: key not found in model")

    if loaded:
        model.load_state_dict(model_state)
        print(f"Partially loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No compatible params found in checkpoint: {checkpoint_path}")
    return loaded

# Define file paths for persistent chat storage
CHAT_MEMORY_FILE = "chat_memory.json"
TRAINING_DATA_FILE = "chat_training_data.json"

#Load conversation
def load_conversation(file_path=CHAT_MEMORY_FILE):
    """Load conversation history from file. Create the file if it doesn't exist."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)
    with open(file_path, "r") as f:
        return json.load(f)

#save conversation
def save_conversation(history, file_path=CHAT_MEMORY_FILE):
    """Save conversation history to a file with pretty-printing."""
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)

#chat_with_ai
def chat_with_bot(model, word_to_index, index_to_word, device, memory_size=5):
    """
    Chat function using tokenized conversation memory.
    The chatbot stores user inputs and bot responses persistently.
    """
    print("\nneural network is running! Type 'exit' to quit.")
    temperature = config.TEMPERATURE
    conversation_history = load_conversation()

    # In-memory list for current session conversation (each element is a dict)
    session_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            # Save the updated conversation to file before quitting
            conversation_history.extend(session_history)
            save_conversation(conversation_history)
            break

        # Tokenize the user input using the data module's preprocessing so
        # the same pipeline is used for chat and training.
        user_tokens = data.preprocess_data(user_input)
        tokenized_input = data.tokenize_and_numericalize(user_tokens, word_to_index)

        # Append user input to session history
        session_history.append({"speaker": "user", "text": user_input, "tokens": tokenized_input})
        # Keep only the last `memory_size` messages in session history for context
        context = session_history[-memory_size:]

        # Prepare context as input for the model by concatenating tokens (convert to a list of ints)
        # Here we simply flatten the token lists. In a more advanced version, you might use special tokens.
        context_tokens = sum([entry["tokens"] for entry in context], [])
        # Convert token list to a string that the inference module can process.
        # This is a simplistic approach; you might instead pass the tokens directly if your model supports it.
        # index_to_word uses integer keys, so lookup should use ints.
        context_text = " ".join([index_to_word.get(token, "<UNK>") for token in context_tokens])

        # Generate a response using the LSTM model (you could choose a different model if desired)
        bot_response = generate_text_simple(
            model, context_text, word_to_index, index_to_word,
            config.NUM_WORDS_TO_GENERATE, temperature, device, model_type="lstm"
        )

        print(f"Chatbot: {bot_response}")
        # Tokenize bot response
        bot_tokens = data.preprocess_data(bot_response)
        tokenized_response = data.tokenize_and_numericalize(bot_tokens, word_to_index)
        # Append bot response to session history
        session_history.append({"speaker": "bot", "text": bot_response, "tokens": tokenized_response})

def main(mode='all', skip_chat=False, epochs_override=None, prompt_override=None, scrape=False):
    """Main entrypoint.

    mode: 'all' (default), 'train', or 'infer'
    skip_chat: if True, do not enter interactive chat
    epochs_override: if provided, override config.EPOCHS for this run
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Initialize PID controller for temperature control.
    pid_temp = PIDController(Kp=0.2, Ki=0.02, Kd=0.01, setpoint=0.8)

    # If data file is empty or user requested scraping, run the data collection pipeline
    try:
        data_exists = os.path.exists(config.DATA_FILE) and os.path.getsize(config.DATA_FILE) > 0
    except Exception:
        data_exists = False

    if not data_exists or scrape:
        if scrape:
            print("--scrape flag provided; forcing data collection pipeline...")
        else:
            print("Data file empty or missing — running data collection pipeline...")
        try:
            data.setup_directories()
            # Use a couple of sensible defaults; these functions are safe if optional deps are missing
            # Pass a sensible user agent from config so wikipedia scraping is allowed
            data.fetch_wikipedia_articles([
    "Greetings",  
    "Etiquette",  
    "Philosophy",  
    "Science",  
    "Technology",  
    "History",  
    "Geography",  
    "Mathematics",  
    "Literature",  
    "Art",  
    "Music",  
    "Linguistics",  
    "Politics",  
    "Law",  
    "Economics",  
    "Psychology",  
    "Sociology",  
    "Religion",  
    "Mythology",  
    "Anthropology",  
    "Biology",  
    "Physics",  
    "Chemistry",  
    "Computer Science",  
    "Artificial Intelligence",  
    "Medicine",  
    "Engineering",  
    "Astronomy",  
    "Environmental Science",  
    "Education",  
    "Cultural Studies",  
    "Sports",  
    "Games",  
    "Food",  
    "Travel",  
    "Fashion",  
    "Film",  
    "Theater",  
    "Architecture"  
])

            data.fetch_live_data(["https://www.gutenberg.org/files/1342/1342-0.txt"]) 
            data.compile_data()
            print("Data collection complete.")
        except Exception as e:
            print(f"Data collection pipeline failed: {e}")

    # Data Loading and Preprocessing
    text = data.load_data(config.DATA_FILE)
    tokens = data.preprocess_data(text)

    # Data Augmentation
    if config.USE_AUGMENTATION:
        augmented_tokens = augmentation.augment_data(tokens)
        # Choose one augmented version to replace the original tokens.
        tokens = augmented_tokens[0]

    word_to_index, index_to_word = data.build_vocabulary(tokens, config.MIN_FREQUENCY)
    numericalized_tokens = data.tokenize_and_numericalize(tokens, word_to_index)
    sequences = data.create_sequences(numericalized_tokens, config.SEQUENCE_LENGTH)
    train_seqs, val_seqs = data.split_data(sequences, config.SPLIT_RATIO)
    train_inputs, train_targets = data.data_to_tensors(train_seqs, device)
    val_inputs, val_targets = data.data_to_tensors(val_seqs, device)

    vocab_size = len(word_to_index)

    # Ensure MLP output size covers the actual target token indices. Sometimes
    # the observed max token id in train/val may exceed the nominal vocab_size
    # (due to preprocessing differences). Compute a safe output size.
    mlp_output_size = vocab_size
    try:
        if 'train_targets' in locals() and hasattr(train_targets, 'numel') and train_targets.numel() > 0:
            observed_max = int(torch.max(train_targets).item())
            mlp_output_size = max(vocab_size, observed_max + 1)
        elif 'val_targets' in locals() and hasattr(val_targets, 'numel') and val_targets.numel() > 0:
            observed_max = int(torch.max(val_targets).item())
            mlp_output_size = max(vocab_size, observed_max + 1)
    except Exception:
        mlp_output_size = vocab_size

    # Model Initialization with Ensemble (including multiple architectures)
    models = {
        "rnn": model.RNNTextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_SIZE,
                                            config.NUM_LAYERS, config.DROPOUT, config.MULTI_TASK).to(device),
        "gru": model.GRUTextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_SIZE,
                                            config.NUM_LAYERS, config.DROPOUT, config.MULTI_TASK).to(device),
        "lstm": model.LSTMTextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_SIZE,
                                              config.NUM_LAYERS, config.DROPOUT, config.MULTI_TASK).to(device),
        "transformer": model.TransformerTextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.NHEAD,
                                                            config.NUM_LAYERS, config.DROPOUT, config.MULTI_TASK).to(device),
    # For MLP we expect a flattened one-hot input across the sequence length.
    # Build input_size as vocab_size * sequence_length and set output_size to vocab_size
    "mlp": model.MLPModel(vocab_size * config.SEQUENCE_LENGTH,
                          config.MODEL_CONFIG['hidden_layers'],
                          mlp_output_size,
                          config.DROPOUT).to(device)
    }

    # Initialize a meta-optimizer for each model.
    optimizers = {name: MetaOptimizer(m, base_lr=config.LEARNING_RATE,
                                       scheduler_step=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)
                  for name, m in models.items()}

    criterion = torch.nn.CrossEntropyLoss()
    aux_criterion = torch.nn.CrossEntropyLoss() if config.MULTI_TASK else None

    # Determine epochs to use for this run
    epochs_to_run = epochs_override if epochs_override is not None else config.EPOCHS

    # Training with Active Learning and PID control over several rounds
    if mode in ('all', 'train'):
        for iteration in range(config.ACTIVE_LEARNING_ROUNDS):
            print(f"\n=== Active Learning Round {iteration + 1} ===")
            for name, net in models.items():
                print(f"\nTraining model: {name.upper()}")
                loss = train_run(
                    net,
                    train_inputs,
                    train_targets,
                    val_inputs,
                    val_targets,
                    optimizers[name],
                    criterion,
                    aux_criterion,
                    epochs_to_run,
                    config.BATCH_SIZE,
                    device,
                    clip=config.CLIP,
                    early_stop_patience=config.EARLY_STOP_PATIENCE,
                    checkpoint_path=f"saved_models/best_model_{name}.pth",
                    model_type=name,
                    multi_task=config.MULTI_TASK,
                    writer=writer,
                )
                # Adjust learning rate using PID for each model.
                # train_run may perform optimizer steps internally and returns a scalar loss.
                # Only call MetaOptimizer.step when a tensor-like loss is returned.
                try:
                    if hasattr(loss, 'backward'):
                        optimizers[name].step(loss)
                    else:
                        # scalar loss (float) — skip calling MetaOptimizer.step
                        pass
                except Exception as e:
                    print(f"Warning: skipping optimizer.step due to: {e}")
                print(f"[{name.upper()}] Updated Learning Rate: {optimizers[name].get_lr()}")

            # Load best model checkpoints after training round.
            for name, net in models.items():
                checkpoint_path = f"saved_models/best_model_{name}.pth"
                if os.path.exists(checkpoint_path):
                    try:
                        loaded_any = safe_load_checkpoint(net, checkpoint_path)
                        if loaded_any:
                            print(f"Best {name.upper()} model partially loaded from checkpoint.")
                        else:
                            print(f"Best {name.upper()} model checkpoint present but incompatible; skipping.")
                    except Exception as e:
                        print(f"Warning: failed to load checkpoint for {name}: {e}")

            # Active Learning: Select uncertain samples (using LSTM as representative) and augment.
            if config.USE_ACTIVE_LEARNING:
                uncertain_samples = active_learning.select_uncertain_samples(models["lstm"], train_inputs,
                                                                             config.ACTIVE_LEARNING_TOP_K, device)
                augmented_samples = augmentation.augment_data(uncertain_samples)
                new_inputs, new_targets = data.data_to_tensors(augmented_samples, device)
                train_inputs = torch.cat([train_inputs, new_inputs])
                train_targets = torch.cat([train_targets, new_targets])

    # Inference with Ensemble Voting
    ensemble_outputs = []
    if mode in ('all', 'infer'):
        # If a prompt_override is provided, convert it using the same preprocessing
        # pipeline used during training/chat so the model receives token-aligned input.
        if prompt_override is not None:
            # Preprocess the prompt into token strings (same pipeline used for training/chat)
            prompt_tokens = data.preprocess_data(prompt_override)
            # Convert tokens to ids (pass these ids directly to the generator)
            start_words = data.tokenize_and_numericalize(prompt_tokens, word_to_index)
            token_mode = True
        else:
            start_words = config.START_WORDS
            token_mode = False

        for name, net in models.items():
            output_text = generate_text_simple(
                net,
                start_words,
                word_to_index,
                index_to_word,
                config.NUM_WORDS_TO_GENERATE,
                config.TEMPERATURE,
                device,
                model_type=name,
            )
            ensemble_outputs.append((output_text, 1))  # Equal weight for now

    final_output = utils.weighted_ensemble_output(ensemble_outputs)
    
    # Adaptive Temperature Adjustment using PID
    temperature = config.TEMPERATURE
    for _ in range(3):
        # Use the same preprocessed start words if a prompt_override is provided
        if prompt_override is not None:
            temp_prompt_tokens = data.preprocess_data(prompt_override)
            temp_start = data.tokenize_and_numericalize(temp_prompt_tokens, word_to_index)
        else:
            temp_start = config.START_WORDS

        generated = generate_text_simple(
            models["lstm"],
            temp_start,
            word_to_index,
            index_to_word,
            config.NUM_WORDS_TO_GENERATE,
            temperature,
            device,
            model_type="lstm",
        )
        temperature += pid_temp.update(len(generated.split()) / config.NUM_WORDS_TO_GENERATE)
        temperature = max(0.5, min(1.5, temperature))
        print(f"\nGenerated Text at Temperature {temperature}:\n{generated}")

    print("\nEnsemble Generated Text:\n", final_output)
    
    writer.close()

    # Start the interactive chat session after building/loading models.
    # If prompt_override was provided, we still allow entering chat when --chat-only
    # was requested; otherwise, respect skip_chat and mode.
    if token_mode:
        # If the user provided a prompt and didn't request chat-only, exit after generation
        if not args.chat_only:
            print("Prompt provided; skipping interactive chat and exiting.")
            return

    # If chat-only was requested, always enter chat (after attempting to load checkpoints)
        if args.chat_only:
            # Attempt to load checkpoints if present (using safe loader)
            for name, net in models.items():
                checkpoint_path = f"saved_models/best_model_{name}.pth"
                if os.path.exists(checkpoint_path):
                    try:
                        safe_load_checkpoint(net, checkpoint_path)
                    except Exception as e:
                        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            # Enter interactive chat
            chat_with_bot(models["lstm"], word_to_index, index_to_word, device)
            return

    # Honor skip_chat flag for non-interactive runs.
    if not skip_chat and mode in ('all',):
        chat_with_bot(models["lstm"], word_to_index, index_to_word, device)
    else:
        print("Skipping interactive chat and exiting.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run training and/or inference for the neural network project')
    parser.add_argument('--mode', choices=['all', 'train', 'infer'], default='all', help='Run mode: all (train+infer), train, or infer')
    parser.add_argument('--skip-chat', action='store_true', help='Skip interactive chat and exit after inference')
    parser.add_argument('--prompt', type=str, help='Provide a prompt string to generate text once and exit (implies --mode infer)')
    parser.add_argument('--epochs', type=int, help='Temporarily override number of epochs for this run')
    parser.add_argument('--interactive', action='store_true', help='Start an interactive menu to choose run options')
    parser.add_argument('--chat-only', action='store_true', help='Skip training/inference and enter chat (loads saved checkpoints if available)')
    parser.add_argument('--scrape', action='store_true', help='Force scraping/data collection pipeline even if data file exists')
    args = parser.parse_args()

    # If a prompt is provided, prefer inference and skip entering the interactive chat.
    if args.prompt is not None and args.mode == 'all':
        args.mode = 'infer'
        args.skip_chat = True

    def interactive_menu():
        """Simple text menu to pick an action and run main accordingly."""
        menu = """
Interactive menu - pick an option:
1) Train only
2) Infer only
3) Train + Infer + Chat (default 'all')
4) Chat only (loads models and enters chat)
5) Exit
"""
        while True:
            print(menu)
            choice = input("Select (1-5): ").strip()
            if choice == '1':
                epochs = input("Epochs (press Enter for default): ").strip()
                epochs = int(epochs) if epochs else None
                main(mode='train', skip_chat=True, epochs_override=epochs)
            elif choice == '2':
                main(mode='infer', skip_chat=True)
            elif choice == '3':
                epochs = input("Epochs (press Enter for default): ").strip()
                epochs = int(epochs) if epochs else None
                main(mode='all', skip_chat=False, epochs_override=epochs)
            elif choice == '4':
                # Chat only: run main but skip training/inference by using infer and then entering chat.
                # For simplicity call main in 'all' mode without skipping chat so it will build models then chat.
                main(mode='all', skip_chat=False)
            elif choice == '5':
                print("Exiting interactive menu.")
                break
            else:
                print("Unknown selection, try again.")

    if args.interactive:
        interactive_menu()
    else:
        # If chat-only was requested, call main with mode set to 'infer' but skip the ensemble generation
        if args.chat_only:
            # main will build models and then enter chat when skip_chat=False
            main(mode='infer', skip_chat=False, epochs_override=args.epochs, prompt_override=None, scrape=args.scrape)
        else:
            main(mode=args.mode, skip_chat=args.skip_chat, epochs_override=args.epochs, prompt_override=args.prompt, scrape=args.scrape)

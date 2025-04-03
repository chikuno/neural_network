
#Module imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json

# Import custom modules
import config
import data
import model
import train
import inference
import utils
import augmentation  
import active_learning  
from pid_controller import PIDController  
from train.optimizer import MetaOptimizer

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

        # Tokenize the user input using the data module
        # (Assuming data.tokenize_and_numericalize accepts a list of strings and returns list of token lists)
        tokenized_input = data.tokenize_and_numericalize([user_input], word_to_index)[0]

        # Append user input to session history
        session_history.append({"speaker": "user", "text": user_input, "tokens": tokenized_input})
        # Keep only the last `memory_size` messages in session history for context
        context = session_history[-memory_size:]

        # Prepare context as input for the model by concatenating tokens (convert to a list of ints)
        # Here we simply flatten the token lists. In a more advanced version, you might use special tokens.
        context_tokens = sum([entry["tokens"] for entry in context], [])
        # Convert token list to a string that the inference module can process.
        # This is a simplistic approach; you might instead pass the tokens directly if your model supports it.
        context_text = " ".join([index_to_word.get(str(token), "<UNK>") for token in context_tokens])

        # Generate a response using the LSTM model (you could choose a different model if desired)
        bot_response = inference.generate_text_simple(
            model, context_text, word_to_index, index_to_word,
            config.NUM_WORDS_TO_GENERATE, temperature, device, model_type="lstm"
        )

        print(f"Chatbot: {bot_response}")
        # Tokenize bot response
        tokenized_response = data.tokenize_and_numericalize([bot_response], word_to_index)[0]
        # Append bot response to session history
        session_history.append({"speaker": "bot", "text": bot_response, "tokens": tokenized_response})

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Initialize PID controllers for learning rate and temperature control.
    pid_lr = PIDController(Kp=0.1, Ki=0.01, Kd=0.01, setpoint=0.02)
    pid_temp = PIDController(Kp=0.2, Ki=0.02, Kd=0.01, setpoint=0.8)

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
        "mlp": model.MLPTextGenerationModel(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_SIZE, config.DROPOUT).to(device)
    }

    # Initialize a meta-optimizer for each model.
    optimizers = {name: MetaOptimizer(m, base_lr=config.LEARNING_RATE,
                                       scheduler_step=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)
                  for name, m in models.items()}

    criterion = torch.nn.CrossEntropyLoss()
    aux_criterion = torch.nn.CrossEntropyLoss() if config.MULTI_TASK else None

    # Training with Active Learning and PID control over several rounds
    for iteration in range(config.ACTIVE_LEARNING_ROUNDS):
        print(f"\n=== Active Learning Round {iteration + 1} ===")
        for name, net in models.items():
            print(f"\nTraining model: {name.upper()}")
            loss = train.train(net, train_inputs, train_targets, val_inputs, val_targets,
                               optimizers[name], criterion, aux_criterion, config.EPOCHS,
                               config.BATCH_SIZE, device, clip=config.CLIP,
                               early_stop_patience=config.EARLY_STOP_PATIENCE,
                               checkpoint_path=f"saved_models/best_model_{name}.pth",
                               model_type=name, multi_task=config.MULTI_TASK, writer=writer)
            # Adjust learning rate using PID for each model.
            optimizers[name].step(loss)
            print(f"[{name.upper()}] Updated Learning Rate: {optimizers[name].get_lr()}")

        # Load best model checkpoints after training round.
        for name, net in models.items():
            checkpoint_path = f"saved_models/best_model_{name}.pth"
            if os.path.exists(checkpoint_path):
                net.load_state_dict(torch.load(checkpoint_path))
                print(f"Best {name.upper()} model loaded.")

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
    for name, net in models.items():
        output_text = inference.generate_text_simple(net, config.START_WORDS, word_to_index, index_to_word,
                                                     config.NUM_WORDS_TO_GENERATE, config.TEMPERATURE,
                                                     device, model_type=name)
        ensemble_outputs.append((output_text, 1))  # Equal weight for now

    final_output = utils.weighted_ensemble_output(ensemble_outputs)
    
    # Adaptive Temperature Adjustment using PID
    temperature = config.TEMPERATURE
    for _ in range(3):
        generated = inference.generate_text_simple(models["lstm"], config.START_WORDS, word_to_index, index_to_word,
                                                   config.NUM_WORDS_TO_GENERATE, temperature, device,
                                                   model_type="lstm")
        temperature += pid_temp.update(len(generated.split()) / config.NUM_WORDS_TO_GENERATE)
        temperature = max(0.5, min(1.5, temperature))
        print(f"\nGenerated Text at Temperature {temperature}:\n{generated}")

    print("\nEnsemble Generated Text:\n", final_output)
    
    writer.close()

    # Start the interactive chat session after training and inference.
    # Here we use the LSTM model for chatting.
    chat_with_bot(models["lstm"], word_to_index, index_to_word, device)

if __name__ == '__main__':
    main()

# config.py

# ----------------------------
# Data & File Paths
# ----------------------------

DATA_FILE = "data/data.txt"           # Path to the merged training data file
CHECKPOINT_PATH = "saved_models/best_model.pth"  # Default path to save best model
LOG_DIR = "logs"                      # Directory to store logs and TensorBoard outputs


# ----------------------------
# General Parameters
# ----------------------------

SEQUENCE_LENGTH = 30                 # Sequence length for training
MIN_FREQUENCY = 5                    # Minimum frequency of words to consider
BATCH_SIZE = 64                      # Batch size for training
EPOCHS = 5                           # Number of training epochs per active learning round
LEARNING_RATE = 0.001                # Base learning rate
DROPOUT = 0.3                        # Dropout rate to prevent overfitting
SPLIT_RATIO = 0.8                    # Train/validation split ratio


# ----------------------------
# Embedding & Hidden Dimensions
# ----------------------------

EMBEDDING_DIM = 256                  # Dimension of the word embeddings
HIDDEN_SIZE = 512                    # Hidden layer size for recurrent/transformer models
NUM_LAYERS = 2                       # Number of layers in the model


# ----------------------------
# Scheduler & Training Enhancements
# ----------------------------

SCHEDULER_STEP = 3                   # Step size for learning rate scheduler
SCHEDULER_GAMMA = 0.5                # Decay factor for learning rate scheduler
CLIP = 5                             # Maximum gradient norm for gradient clipping
EARLY_STOP_PATIENCE = 5


# ----------------------------
# Generation Settings
# ----------------------------

START_WORDS = ["the", "quick", "brown"]  # Initial seed words for text generation
NUM_WORDS_TO_GENERATE = 100              # Number of words to generate during inference
TEMPERATURE = 1.0                        # Initial sampling temperature for text generation


# ----------------------------
# Multi-task Learning
# ----------------------------

MULTI_TASK = False                   # Enable multi-task learning (auxiliary tasks)


# ----------------------------
# Model Type Selection
# Options: 'rnn', 'gru', 'lstm', 'transformer'
# ----------------------------

MODEL_TYPE = "lstm"                  # Selected model type


# ----------------------------
# Transformer Specific
# ----------------------------

NHEAD = 4                            # Number of attention heads (for transformer model)


# ----------------------------
# Data Augmentation & Active Learning
# ----------------------------

USE_AUGMENTATION = True              # Enable data augmentation methods
USE_ACTIVE_LEARNING = True           # Enable active learning loops
ACTIVE_LEARNING_ROUNDS = 2           # Number of active learning iterations
ACTIVE_LEARNING_TOP_K = 5            # Number of uncertain samples to augment


# ----------------------------
# Meta-Learning Settings
# ----------------------------

META_LEARNING = False                # Enable meta-learning (e.g., MAML)
META_LEARNING_RATE = 0.0005          # Learning rate for meta-learning updates


# ----------------------------
# Ensemble Settings
# ----------------------------

USE_ENSEMBLE = True                  # Enable ensemble methods for text generation


# ----------------------------
# Model Configuration (Neural Network Architecture)
# ----------------------------

MODEL_CONFIG = {
    'input_size': 784,               # Size of the input layer (e.g., for 28x28 images, input_size = 784)
    'hidden_layers': [512, 256, 128], # List of sizes for each hidden layer
    'output_size': 10,                # Size of the output layer (e.g., for 10-class classification)
    'task_outputs': [10, 1],          # Multi-task outputs: primary output (e.g., classification) and auxiliary task (e.g., regression)
    'dropout': 0.5,                   # Dropout rate to prevent overfitting
    'activation': 'relu',             # Activation function (e.g., 'relu' or 'sigmoid')
    'weight_decay': 1e-4              # Weight decay for regularization (L2 penalty)
}

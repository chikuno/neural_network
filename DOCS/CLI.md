CLI Flags

--prompt "TEXT"
  - Provide a one-off prompt string. The prompt is preprocessed using the same pipeline as training/chat, tokenized, and passed directly into the model for generation. When provided, the run defaults to inference mode and will exit after generation unless --chat-only is also used.

--chat-only
  - Build models (and attempt to load checkpoints from saved_models/best_model_<model>.pth if present) and then enter the interactive chat prompt. This skips the full training/inference ensemble flow and is useful for quick interactive testing.

--scrape
  - Force the data collection pipeline (scraping and compiling) even if a data file already exists. Useful to refresh the dataset or gather new training material.

--random-prompt
  - Ignore --prompt and seed generation with a random window sampled from the training tokens.

--no-refine
  - Disable post-processing refinement of generated text.

--no-memory
  - Disable use of chat memory context during generation/chat.

--model [rnn|gru|lstm|transformer|mlp]
  - Select a single model for generation instead of using an ensemble.

--no-active-learning
  - Disable active learning augmentation during training.

--max-train-batches N
  - Limit the number of training batches per epoch for quick smoke tests.

--device [auto|cpu|cuda]
  - Select compute device. 'auto' picks cuda if available, otherwise cpu.

--no-persona
  - Disable persona prefixing in chat/inference. By default, the bot conditions on its identity.

--decoding [sampling|beam]
  - Choose decoding strategy. Default is sampling with top-k/top-p and repetition controls.

--beam-size N
  - Beam width when using beam decoding.

--length-penalty P
  - Length normalization penalty used in beam scoring. 0 = disabled, typical 0.6–1.0.

--early-stop / --no-early-stop
  - Toggle early stopping on EOS token during generation (requires --eos-token to be in vocab).

--min-length N
  - Minimum number of generated tokens before EOS is allowed.

--eos-token TOKEN
  - Token string to treat as EOS (must exist in the vocabulary, e.g., '<EOS>').

--seed N
  - Set a global RNG seed for reproducible sampling and random prompt selection.

Examples (scraping)

# Force scraping and run inference (non-interactive)
python main.py --mode infer --skip-chat --scrape

# Force scraping and then enter chat-only mode
python main.py --chat-only --scrape

Examples

# One-off generation with a prompt (non-interactive)
python main.py --prompt "Hello world"

# Deterministic run with beam search
python main.py --mode infer --skip-chat --decoding beam --beam-size 4 --length-penalty 0.7 --eos-token '<EOS>' --early-stop --min-length 10 --seed 42

# Start chat only (loads checkpoints if available)
python main.py --chat-only

# Identity questions
# The bot answers deterministically for identity prompts
python main.py --prompt "Who are you?"

CLI Flags

--prompt "TEXT"
  - Provide a one-off prompt string. The prompt is preprocessed using the same pipeline as training/chat, tokenized, and passed directly into the model for generation. When provided, the run defaults to inference mode and will exit after generation unless --chat-only is also used.

--chat-only
  - Build models (and attempt to load checkpoints from saved_models/best_model_<model>.pth if present) and then enter the interactive chat prompt. This skips the full training/inference ensemble flow and is useful for quick interactive testing.

--scrape
  - Force the data collection pipeline (scraping and compiling) even if a data file already exists. Useful to refresh the dataset or gather new training material.

Examples (scraping)

# Force scraping and run inference (non-interactive)
python main.py --mode infer --skip-chat --scrape

# Force scraping and then enter chat-only mode
python main.py --chat-only --scrape

Examples

# One-off generation with a prompt (non-interactive)
python main.py --prompt "Hello world"

# Start chat only (loads checkpoints if available)
python main.py --chat-only

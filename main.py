#Module imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import json

# Import custom modules
from config import config
import data.data as data
from train.train import train as train_run
import model
# 'train' and 'inference' modules imported implicitly via specific imports above
from inference.generate import generate_text_simple, generate_stream
# from inference.generate import generate_text_learned_ensemble  # no longer used; ActiveRunner handles ensemble
from inference.refine import refine_text
import utils
import data.augmentation as augmentation
import data.active_learning as active_learning
from model.pid_controller import PIDController
import numpy as np
from train.optimizer import MetaOptimizer
from train.class_weights import compute_class_weights
from scripts.active_runner import get_runner


def safe_load_checkpoint(model, checkpoint_path):
    """Attempt to load checkpoint into model but only copy parameters that match in name and shape.
    Prints a summary of loaded/skipped keys and returns True if any keys were loaded.
    """
    if not os.path.exists(checkpoint_path):
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to read checkpoint {checkpoint_path}: {e}")
        return False

    model_state = model.state_dict()
    # Accept both pure state_dicts and wrapped dicts with a 'model' key
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt_state = ckpt['model']
    else:
        ckpt_state = ckpt if isinstance(ckpt, dict) else {}
    loaded = False
    for k, v in ckpt_state.items():
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
CHAT_LONG_TERM_FILE = "chat_memory_long_term.json"
TRAINING_DATA_FILE = "chat_training_data.json"

#Load conversation
def load_conversation(file_path=CHAT_MEMORY_FILE):
    """Load conversation history from file. Create the file if it doesn't exist."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)
    with open(file_path, "r") as f:
        return json.load(f)

def load_long_term_memory(file_path=CHAT_LONG_TERM_FILE):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({"facts": []}, f)
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and 'facts' in data:
                return data
            return {"facts": []}
        except Exception:
            return {"facts": []}

def save_long_term_memory(mem, file_path=CHAT_LONG_TERM_FILE):
    try:
        with open(file_path, "w") as f:
            json.dump(mem, f, indent=4)
    except Exception:
        pass

#save conversation
def save_conversation(history, file_path=CHAT_MEMORY_FILE):
    """Save conversation history to a file with pretty-printing."""
    with open(file_path, "w") as f:
        json.dump(history, f, indent=4)

#chat_with_ai
def chat_with_bot(model, word_to_index, index_to_word, device, memory_size=None):
    """
    Chat function using tokenized conversation memory.
    The chatbot stores user inputs and bot responses persistently.
    """
    print("\nneural network is running! Type 'exit' to quit.")
    temperature = config.TEMPERATURE
    conversation_history = load_conversation()
    long_term = load_long_term_memory()

    # In-memory list for current session conversation (each element is a dict)
    session_history = []
    # Pull chat UX defaults
    if memory_size is None:
        memory_size = getattr(args, 'memory_size', None) if 'args' in globals() else None
    if memory_size is None:
        memory_size = getattr(config, 'CHAT_MEMORY_SIZE', 6)
    style = getattr(args, 'style', None) if 'args' in globals() else None
    if style is None:
        style = getattr(config, 'CHAT_STYLE', 'friendly')
    do_stream = bool(getattr(args, 'stream', False)) if 'args' in globals() and getattr(args, 'stream', False) else bool(getattr(config, 'CHAT_STREAM', False))
    typing_indicator = bool(getattr(config, 'CHAT_TYPING_INDICATOR', True))
    safe_mode = bool(getattr(config, 'SAFE_MODE', True))
    memory_mode = getattr(args, 'memory_mode', None) if 'args' in globals() else None
    if memory_mode is None:
        memory_mode = getattr(config, 'MEMORY_MODE', 'session')

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            # Save the updated conversation to file before quitting
            conversation_history.extend(session_history)
            save_conversation(conversation_history)
            break

        # Tooling: simple real-world functions
        if getattr(config, 'ENABLE_TOOLS', False):
            low = user_input.lower()
            if low.strip() in ("help tools", "tools help", "help: tools"):
                msg = (
                    "Tools available:\n"
                    "- Time/Date: ask 'what time is it' or 'what day is it'\n"
                    "- Calculator: 'calc <expr>' (numbers and + - * / ( ))\n"
                    "- Converter: 'convert <value> <unit> to <unit>' (km<->mi, kg<->lb, c<->f)"
                )
                print(f"Chatbot: {msg}")
                session_history.append({"speaker": "bot", "text": msg, "tokens": []})
                continue
            # Time/date
            if getattr(config, 'TOOLS_TIME', True) and any(k in low for k in ["time", "date", "today", "day is it"]):
                try:
                    import datetime
                    now = datetime.datetime.now()
                    msg = f"It's {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
                    print(f"Chatbot: {msg}")
                    session_history.append({"speaker": "bot", "text": msg, "tokens": []})
                    continue
                except Exception:
                    pass
            # Calculator (safe eval limited to numbers and ops)
            if getattr(config, 'TOOLS_CALCULATOR', True) and low.startswith("calc "):
                expr = user_input[5:].strip()
                import re
                if re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", expr):
                    try:
                        # Use eval with restricted globals/locals
                        result = eval(expr, {"__builtins__": None}, {})
                        msg = f"Result: {result}"
                        print(f"Chatbot: {msg}")
                        session_history.append({"speaker": "bot", "text": msg, "tokens": []})
                        continue
                    except Exception:
                        pass
            # Unit converter (very small set: km<->mi, kg<->lb, c<->f)
            if getattr(config, 'TOOLS_UNIT_CONVERTER', True) and low.startswith("convert "):
                try:
                    parts = user_input.split()
                    # expected: convert <value> <unit> to <unit>
                    if len(parts) >= 5 and parts[3].lower() == 'to':
                        val = float(parts[1])
                        src = parts[2].lower()
                        dst = parts[4].lower()
                        converted = None
                        if src in ("km","kilometers") and dst in ("mi","miles"):
                            converted = val * 0.621371
                        elif src in ("mi","miles") and dst in ("km","kilometers"):
                            converted = val / 0.621371
                        elif src in ("kg","kilograms") and dst in ("lb","pounds"):
                            converted = val * 2.20462
                        elif src in ("lb","pounds") and dst in ("kg","kilograms"):
                            converted = val / 2.20462
                        elif src in ("c","celsius") and dst in ("f","fahrenheit"):
                            converted = (val * 9/5) + 32
                        elif src in ("f","fahrenheit") and dst in ("c","celsius"):
                            converted = (val - 32) * 5/9
                        if converted is not None:
                            msg = f"{val} {src} = {converted:.4g} {dst}"
                            print(f"Chatbot: {msg}")
                            session_history.append({"speaker": "bot", "text": msg, "tokens": []})
                            continue
                except Exception:
                    pass

        # Tokenize the user input using the data module's preprocessing so
        # the same pipeline is used for chat and training.
        user_tokens = data.preprocess_data(user_input)
        tokenized_input = data.tokenize_and_numericalize(user_tokens, word_to_index)

        # Append user input to session history
        session_history.append({"speaker": "user", "text": user_input, "tokens": tokenized_input})
        # Keep only the last `memory_size` messages in session history for context
        context = session_history[-memory_size:]
        # Long-term memory summary
        lt_facts = long_term.get('facts', [])
        lt_summary = ""
        if memory_mode in ('full','summary') and lt_facts:
            if memory_mode == 'full':
                lt_summary = "Known facts: " + "; ".join(lt_facts) + ". "
            else:
                # summary mode: take last few facts
                tail = lt_facts[-5:]
                lt_summary = "Known facts: " + "; ".join(tail) + ". "

        # If the user asks about identity, answer deterministically
        lowered = user_input.lower()
        identity_prompts = [
            "who are you", "what is your name", "who made you", "who created you", "what are you"
        ]
        if any(q in lowered for q in identity_prompts):
            identity_answer = f"I am {config.BOT_NAME}, an AI assistant created by {config.BOT_CREATOR}."
            print(f"Chatbot: {identity_answer}")
            # Store response and continue
            bot_tokens = data.preprocess_data(identity_answer)
            tokenized_response = data.tokenize_and_numericalize(bot_tokens, word_to_index)
            session_history.append({"speaker": "bot", "text": identity_answer, "tokens": tokenized_response})
            continue

        # Memory UX commands
        if lowered.strip() in ("what do you remember about me?", "what do you remember about me", "what do you remember?"):
            facts = long_term.get('facts', [])
            if facts:
                pretty = []
                for f in facts[-20:]:
                    pretty.append("- " + f.replace('_', ' '))
                msg = "Here are some things I remember about you:\n" + "\n".join(pretty)
            else:
                msg = "I don't have any saved facts yet."
            print(f"Chatbot: {msg}")
            session_history.append({"speaker": "bot", "text": msg, "tokens": []})
            continue
        if lowered.startswith("forget "):
            key = user_input.strip()[7:].strip()
            removed = False
            if key:
                facts = long_term.get('facts', [])
                new_facts = [f for f in facts if key not in f]
                if len(new_facts) != len(facts):
                    long_term['facts'] = new_facts
                    save_long_term_memory(long_term)
                    removed = True
            msg = "Okay, I’ve forgotten that." if removed else "I couldn't find that in memory."
            print(f"Chatbot: {msg}")
            session_history.append({"speaker": "bot", "text": msg, "tokens": []})
            continue

        # Basic safety refusal for harmful content if enabled
        if safe_mode:
            refusals = [
                "Sorry, I can't assist with that.",
            ]
            bad_phrases = [
                'harm yourself', 'violence', 'kill', 'illegal', 'hate speech'
            ]
            if any(p in lowered for p in bad_phrases):
                resp = refusals[0]
                print(f"Chatbot: {resp}")
                session_history.append({"speaker": "bot", "text": resp, "tokens": []})
                continue

        # Prepare context as input for the model by concatenating tokens (convert to a list of ints)
        # Here we simply flatten the token lists. In a more advanced version, you might use special tokens.
        context_tokens = sum([entry["tokens"] for entry in context], [])
        # Convert token list to a string that the inference module can process.
        # Prefix persona unless disabled via CLI flag
        context_text = " ".join([index_to_word.get(token, "<UNK>") for token in context_tokens])
        try:
            no_persona = getattr(args, 'no_persona', False)
        except NameError:
            no_persona = False
        style_prefix = ""
        if style == 'friendly':
            style_prefix = "Please respond in a friendly and helpful tone. "
        elif style == 'concise':
            style_prefix = "Be concise: answer briefly and clearly. "
        elif style == 'formal':
            style_prefix = "Use a formal and professional tone. "
        preface = lt_summary + style_prefix
        if not no_persona and config.PERSONA_PREFIX:
            context_text = f"{config.PERSONA_PREFIX} " + preface + context_text
        else:
            context_text = preface + context_text

        # Generate a response using the LSTM model (you could choose a different model if desired)
        if do_stream:
            if typing_indicator:
                print("Chatbot is typing...", end="\r")
            chunks = []
            for token in generate_stream(
                model, context_text, word_to_index, index_to_word,
                config.NUM_WORDS_TO_GENERATE, temperature, device, model_type="lstm"
            ):
                chunks.append(token)
                print("Chatbot: " + " ".join(chunks), end="\r")
            print()  # newline after stream
            bot_response = " ".join(chunks)
        else:
            bot_response = generate_text_simple(
                model, context_text, word_to_index, index_to_word,
                config.NUM_WORDS_TO_GENERATE, temperature, device, model_type="lstm"
            )

        # Refine the response text before printing
        bot_response = refine_text(bot_response)
        print(f"Chatbot: {bot_response}")
        # Tokenize bot response
        bot_tokens = data.preprocess_data(bot_response)
        tokenized_response = data.tokenize_and_numericalize(bot_tokens, word_to_index)
        # Append bot response to session history
        session_history.append({"speaker": "bot", "text": bot_response, "tokens": tokenized_response})

        # Extract simple facts from user input and bot response and persist
        # Heuristic: sentences like "my name is X", "I like Y", "my favorite Z is Y"
        def extract_facts(text):
            t = text.lower()
            facts = []
            # name
            if "my name is" in t:
                try:
                    after = text.split("my name is", 1)[1].strip()
                    name = after.split()[0].strip(',.;:!')
                    if name:
                        facts.append(f"user_name={name}")
                except Exception:
                    pass
            # like
            if "i like" in t:
                try:
                    after = text.split("i like", 1)[1].strip()
                    like = after.split('.')[0].strip()
                    if like:
                        facts.append(f"likes={like}")
                except Exception:
                    pass
            # favorite
            if "my favorite" in t and " is " in t:
                try:
                    seg = text.lower().split("my favorite",1)[1]
                    cat, val = seg.split(" is ",1)
                    cat = cat.strip().replace(' ','_')
                    val = val.split('.')[0].strip()
                    if cat and val:
                        facts.append(f"favorite_{cat}={val}")
                except Exception:
                    pass
            return facts

        new_facts = []
        new_facts += extract_facts(user_input)
        new_facts += extract_facts(bot_response)
        if new_facts:
            # Deduplicate and cap memory to reasonable length
            existing = set(long_term.get('facts', []))
            for f in new_facts:
                if f not in existing:
                    long_term.setdefault('facts', []).append(f)
                    existing.add(f)
            # Keep only last 200 facts
            if len(long_term['facts']) > 200:
                long_term['facts'] = long_term['facts'][-200:]
            save_long_term_memory(long_term)

def main(mode='all', skip_chat=False, epochs_override=None, prompt_override=None, scrape=False):
    """Main entrypoint.

    mode: 'all' (default), 'train', or 'infer'
    skip_chat: if True, do not enter interactive chat
    epochs_override: if provided, override config.EPOCHS for this run
    """
    # Handle global RNG seed if provided
    try:
        user_seed = getattr(args, 'seed', None)
    except NameError:
        user_seed = None
    if user_seed is not None:
        try:
            import random
            random.seed(user_seed)
            torch.manual_seed(user_seed)
            np.random.seed(user_seed)
        except Exception:
            pass
    # Deterministic mode if requested (config or CLI)
    try:
        det_flag = getattr(args, 'deterministic', False)
    except NameError:
        det_flag = False
    if det_flag or getattr(config, 'DETERMINISTIC', False):
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Resolve device preference (args is defined in __main__ but may not be here; guard it)
    try:
        requested_device = getattr(args, 'device', 'auto')
    except NameError:
        requested_device = 'auto'
    if requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("Requested CUDA but torch.cuda.is_available() is False. Falling back to CPU.")
            device = torch.device('cpu')
    elif requested_device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Diagnostics and perf knobs
    if device.type == 'cuda':
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using device: cuda ({name}, CC {cap[0]}.{cap[1]}, {total_mem:.1f} GB)")
        except Exception:
            print("Using device: cuda")
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            # PyTorch 2.x matmul precision hint
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    else:
        print("Using device: cpu")

    # Generate a hash for the current run's configuration to prevent incompatible resumes
    config_hash = utils.generate_config_hash(config)

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
    # =====================
    # Stage 1: Human Interaction (~200 topics)
    # =====================
    "Greetings", "Politeness",
])


            data.fetch_live_data([
                # Pride and Prejudice (Austen)
                "https://www.gutenberg.org/files/1342/1342-0.txt",
                # Alice’s Adventures in Wonderland (Carroll)
                "https://www.gutenberg.org/files/11/11-0.txt",
                # The Adventures of Sherlock Holmes (Doyle)
                "https://www.gutenberg.org/files/1661/1661-0.txt",
                # The Picture of Dorian Gray (Wilde)
                "https://www.gutenberg.org/files/174/174-0.txt",
                # Frankenstein (Shelley)
                "https://www.gutenberg.org/files/84/84-0.txt",
            ]) 
            data.compile_data()
            print("Data collection complete.")
        except Exception as e:
            print(f"Data collection pipeline failed: {e}")

    # Data Loading and Preprocessing
    text = data.load_data(config.DATA_FILE)
    # If SentencePiece is enabled and model not present, train it from the current corpus
    try:
        if getattr(config, 'USE_SENTENCEPIECE', False):
            import os as _os
            prefix = getattr(config, 'SPM_MODEL_PREFIX', 'data/spm_lm')
            if not _os.path.exists(prefix + '.model'):
                print("Training SentencePiece model from current corpus ...")
                data.train_sentencepiece_model(text)
    except Exception as _e:
        print(f"Warning: SentencePiece training skipped due to: {_e}")
    tokens = data.preprocess_data(text)

    # Data Augmentation
    if config.USE_AUGMENTATION:
        augmented_tokens = augmentation.augment_data(tokens)
        # Choose one augmented version to replace the original tokens.
        tokens = augmented_tokens[0]

    # Try to reuse saved vocab for consistent indices, but rebuild if it's too small
    w2i_saved, i2w_saved = data.try_load_vocab()
    if w2i_saved and i2w_saved and isinstance(w2i_saved, dict) and len(w2i_saved) > 2:
        word_to_index, index_to_word = w2i_saved, i2w_saved
    else:
        print("Vocabulary missing or tiny — rebuilding from current tokens...")
        word_to_index, index_to_word = data.build_vocabulary(tokens, config.MIN_FREQUENCY)
    numericalized_tokens = data.tokenize_and_numericalize(tokens, word_to_index)
    sequences = data.create_sequences(numericalized_tokens, config.SEQUENCE_LENGTH)
    if len(sequences) == 0:
        # Adaptive fallback: shrink effective sequence length until we can form sequences
        orig_len = config.SEQUENCE_LENGTH
        tok_len = len(numericalized_tokens)
        if tok_len <= 1:
            print("Error: Not enough tokens (<=1) after preprocessing to build any training sequence. Check data file or preprocessing.")
            return
        # Try progressively smaller lengths down to 1
        chosen_len = None
        for eff_len in range(min(orig_len, tok_len - 1), 0, -1):
            candidate = data.create_sequences(numericalized_tokens, eff_len)
            if candidate:
                sequences = candidate
                chosen_len = eff_len
                print(f"Warning: Not enough tokens for SEQUENCE_LENGTH={orig_len}. Using shorter effective length {eff_len} (tokens={tok_len}).")
                break
        if len(sequences) == 0:
            print("Error: Failed to build any sequences even after fallback. Aborting run.")
            return
        # Persist override into config for rest of run and write an override file
        if chosen_len is not None and chosen_len != orig_len:
            try:
                config.SEQUENCE_LENGTH = chosen_len
                override_meta = {
                    'original_sequence_length': orig_len,
                    'effective_sequence_length': chosen_len,
                    'token_count': tok_len
                }
                with open('config_override.json', 'w', encoding='utf-8') as fov:
                    json.dump(override_meta, fov, indent=4)
                print("Persisted sequence length override to config_override.json")
            except Exception as e:
                print(f"Warning: failed to persist sequence length override: {e}")
    train_seqs, val_seqs = data.split_data(sequences, config.SPLIT_RATIO)
    # Recall replay: mix in extra sequences derived from JSONL sentences
    try:
        if getattr(config, 'USE_RECALL_REPLAY', False):
            jsonl_path = getattr(config, 'RETRIEVAL_INDEX_PATH', 'data/data.jsonl')
            extra = []
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r', encoding='utf-8') as jf:
                    for line in jf:
                        try:
                            rec = json.loads(line)
                            s = rec.get('text', '')
                            if not s:
                                continue
                            tks = data.preprocess_data(s)
                            ids = data.tokenize_and_numericalize(tks, word_to_index)
                            extra.extend(data.create_sequences(ids, config.SEQUENCE_LENGTH))
                        except Exception:
                            continue
            # sample a portion to add
            if extra:
                import random
                ratio = float(getattr(config, 'RECALL_REPLAY_RATIO', 0.1))
                n_add = max(1, int(len(sequences) * ratio))
                random.shuffle(extra)
                train_seqs.extend(extra[:n_add])
                print(f"Recall replay added {min(n_add, len(extra))} sequences from JSONL")
    except Exception as e:
        print(f"Recall replay skipped due to: {e}")
    if len(train_seqs) == 0:
        # Move at least one sample to train if all ended in val due to shuffle edge case with tiny data
        if len(val_seqs) > 0:
            train_seqs, val_seqs = val_seqs, []
            print("Warning: Train split was empty; reassigned all sequences to training.")

    # Create holdout from validation if configured
    holdout_dataset = None
    holdout_loader = None
    holdout_ratio = float(getattr(config, 'HOLDOUT_RATIO', 0.0) or 0.0)
    if holdout_ratio > 0 and len(val_seqs) > 5:
        import math
        import random
        random.shuffle(val_seqs)
        h_sz = max(1, int(math.floor(len(val_seqs) * holdout_ratio)))
        holdout_seqs = val_seqs[:h_sz]
        val_seqs = val_seqs[h_sz:]
        from data.data import TextDataset
        holdout_dataset = TextDataset(holdout_seqs)
        holdout_loader = DataLoader(holdout_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        # holdout_loader will be evaluated after each epoch inside train loop extension (TODO implement)
        print(f"Holdout set carved: {len(holdout_dataset)} sequences (ratio {holdout_ratio:.2f})")

    # Create Dataset and DataLoader instances
    train_dataset = data.TextDataset(train_seqs)
    val_dataset = data.TextDataset(val_seqs)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    vocab_size = len(word_to_index)

    # Ensure MLP output size covers the actual target token indices. Sometimes
    # the observed max token id in train/val may exceed the nominal vocab_size
    # (due to preprocessing differences). Compute a safe output size.
    mlp_output_size = vocab_size
    try:
        # Check max index from the datasets
        train_max = max(seq[-1] for seq in train_seqs) if train_seqs else 0
        val_max = max(seq[-1] for seq in val_seqs) if val_seqs else 0
        observed_max = max(train_max, val_max)
        mlp_output_size = max(vocab_size, observed_max + 1)
    except Exception:
        mlp_output_size = vocab_size

    # Model Initialization with Ensemble (including multiple architectures)
    t_dropout = getattr(config, 'TRANSFORMER_DROPOUT', config.DROPOUT)
    # Compute scaled capacities for larger tasks (derive from MODEL_SCALE at runtime)
    def _scale_factors():
        ms = getattr(config, 'MODEL_SCALE', 'base')
        nm_map = {'tiny': 0.5, 'small': 0.75, 'base': 1.0, 'medium': 1.5, 'large': 2.0, 'xl': 3.0}
        lm_map = {'tiny': 0.5, 'small': 0.75, 'base': 1.0, 'medium': 1.25, 'large': 1.5, 'xl': 2.0}
        return float(nm_map.get(ms, 1.0)), float(lm_map.get(ms, 1.0))
    try:
        scale_mul, layer_mul = _scale_factors()
    except Exception:
        scale_mul, layer_mul = 1.0, 1.0
    def _scale_dim(x):
        return max(1, int(round(x * scale_mul)))
    def _scale_layers(x):
        base = max(1, int(round(x * layer_mul)))
        return base
    emb_dim = _scale_dim(config.EMBEDDING_DIM)
    hid_size = _scale_dim(config.HIDDEN_SIZE)
    num_layers_scaled = _scale_layers(config.NUM_LAYERS)
    # NHEAD must divide embedding dim; adjust upwards to nearest valid divisor within a small search
    nhead_base = getattr(config, 'NHEAD', 4)
    nhead_scaled = max(1, int(round(nhead_base * layer_mul)))
    # Ensure divisibility
    def _fix_nhead(d_model, nhead):
        nh = max(1, nhead)
        # try nearby values up to +/- 4
        for delta in [0,1,-1,2,-2,3,-3,4,-4]:
            cand = max(1, nh + delta)
            if d_model % cand == 0:
                return cand
        # fallback: clamp to greatest divisor <= nh
        for cand in range(min(nh, d_model), 0, -1):
            if d_model % cand == 0:
                return cand
        return 1
    nhead_final = _fix_nhead(emb_dim, nhead_scaled)
    # Scale MLP hidden layers proportionally
    mlp_hidden = [max(1, int(round(h * scale_mul))) for h in config.MODEL_CONFIG['hidden_layers']]
    try:
        print(f"Model scale: {getattr(config,'MODEL_SCALE','base')} | emb_dim={emb_dim}, hidden={hid_size}, layers={num_layers_scaled}, nhead={nhead_final}")
    except Exception:
        pass
    models = {
    "rnn": model.RNNTextGenerationModel(vocab_size, emb_dim, hid_size,
                        num_layers_scaled, config.DROPOUT, config.MULTI_TASK).to(device),
    "gru": model.GRUTextGenerationModel(vocab_size, emb_dim, hid_size,
                        num_layers_scaled, config.DROPOUT, config.MULTI_TASK).to(device),
    "lstm": model.LSTMTextGenerationModel(vocab_size, emb_dim, hid_size,
                          num_layers_scaled, config.DROPOUT, config.MULTI_TASK).to(device),
    "transformer": model.TransformerTextGenerationModel(vocab_size, emb_dim, nhead_final,
                num_layers_scaled, t_dropout, config.MULTI_TASK).to(device),
    # For MLP, we now use an embedding layer.
    "mlp": model.MLPModel(vocab_size,
                          emb_dim,
                          mlp_hidden,
                          mlp_output_size,
                          config.DROPOUT).to(device)
    }

    # Print parameter counts for visibility
    try:
        def _count_params(m):
            return sum(p.numel() for p in m.parameters())
        total_params = 0
        print("Model parameter counts:")
        for _n, _m in models.items():
            cnt = _count_params(_m)
            total_params += cnt
            print(f"  - {_n}: {cnt:,}")
        print(f"  Total (all models): {total_params:,}")
    except Exception:
        pass

    # Initialize a meta-optimizer for each model.
    warmup_steps = 0
    try:
        warmup_steps = int(getattr(args, 'warmup_steps', config.WARMUP_STEPS))
    except Exception:
        warmup_steps = getattr(config, 'WARMUP_STEPS', 0)
    # Estimate total steps for schedulers that need it (epochs * steps_per_epoch)
    try:
        steps_per_epoch = max(1, len(train_loader))
    except Exception:
        steps_per_epoch = 100
    total_steps_est = (epochs_override if epochs_override is not None else config.EPOCHS) * steps_per_epoch
    # Per-model base LR (e.g., allow slightly higher LR for transformer)
    def base_lr_for(name: str):
        if name == 'transformer':
            return float(getattr(config, 'TRANSFORMER_LR', config.LEARNING_RATE))
        return config.LEARNING_RATE
    # Build MetaOptimizers with graceful fallback for test doubles that have a reduced __init__ signature.
    optimizers = {}
    for _name, _model in models.items():
        try:
            optimizers[_name] = MetaOptimizer(
                _model,
                base_lr=base_lr_for(_name),
                scheduler_step=config.SCHEDULER_STEP,
                gamma=config.SCHEDULER_GAMMA,
                use_pid=True,
                warmup_steps=warmup_steps,
                scheduler_type=getattr(config, 'SCHEDULER_TYPE', 'step'),
                total_steps=total_steps_est,
                weight_decay=getattr(config, 'WEIGHT_DECAY', 0.0),
                use_ema=getattr(config, 'USE_EMA', False),
                ema_decay=getattr(config, 'EMA_DECAY', 0.999),
                ema_eval_only=getattr(config, 'EMA_EVAL_ONLY', True),
                ema_start_step=getattr(config, 'EMA_START_STEP', 0),
            )
        except TypeError:
            # Fallback: legacy/test stub signature or tests without new args
            try:
                optimizers[_name] = MetaOptimizer(
                    _model,
                    base_lr=base_lr_for(_name),
                    scheduler_step=config.SCHEDULER_STEP,
                    gamma=config.SCHEDULER_GAMMA,
                )
            except Exception as e:
                print(f"Warning: failed to construct MetaOptimizer for {_name}: {e}")
                optimizers[_name] = None

    # Optional class weighting and label smoothing for better stability
    class_weight = None
    disable_class_weights = False
    try:
        disable_class_weights = getattr(args, 'no_class_weights', False)
    except NameError:
        disable_class_weights = False
    if getattr(config, 'CLASS_WEIGHTING', False) and not disable_class_weights:
        try:
            class_weight = compute_class_weights(
                train_dataset.sequences,
                val_dataset.sequences,
                vocab_size,
                power=getattr(config, 'CLASS_WEIGHT_POWER', 0.5),
                device=device
            )
        except Exception as e:
            print(f"Warning: failed to compute class weights: {e}")
            class_weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=getattr(config, 'LABEL_SMOOTHING', 0.0))
    aux_criterion = torch.nn.CrossEntropyLoss() if config.MULTI_TASK else None

    # Log class weight stats if available
    try:
        if class_weight is not None and writer is not None:
            with torch.no_grad():
                writer.add_scalar('ClassWeights/mean', float(class_weight.mean().item()), 0)
                writer.add_scalar('ClassWeights/std', float(class_weight.std().item()), 0)
                writer.add_scalar('ClassWeights/min', float(class_weight.min().item()), 0)
                writer.add_scalar('ClassWeights/max', float(class_weight.max().item()), 0)
    except Exception:
        pass

    # Determine epochs to use for this run
    epochs_to_run = epochs_override if epochs_override is not None else config.EPOCHS

    # Helper to align new input sequences to an existing target length
    def _align_seq_len(batch_tensor, target_len, pad_value=0):
        if batch_tensor.dim() != 2:
            return batch_tensor
        cur_len = batch_tensor.size(1)
        if cur_len == target_len:
            return batch_tensor
        if cur_len < target_len:
            pad = torch.full((batch_tensor.size(0), target_len - cur_len), pad_value,
                             dtype=batch_tensor.dtype, device=batch_tensor.device)
            return torch.cat([batch_tensor, pad], dim=1)
        else:
            return batch_tensor[:, :target_len]

    # Training with Active Learning and PID control over several rounds
    if mode in ('all', 'train'):
        selected_model = None
        try:
            selected_model = getattr(args, 'model', None)
        except NameError:
            selected_model = None
        for iteration in range(config.ACTIVE_LEARNING_ROUNDS):
            print(f"\n=== Active Learning Round {iteration + 1} ===")
            iter_items = [(selected_model, models[selected_model])] if selected_model else list(models.items())
            # Resume from last checkpoints if requested
            try:
                do_resume = getattr(args, 'resume', False)
            except NameError:
                do_resume = False
            if do_resume:
                for name, net in iter_items:
                    # Prefer last_ files produced by the train loop
                    last_path_a = f"saved_models/last_model_{name}.pth"
                    last_path_b = f"saved_models/best_model_{name}.pth.last"
                    meta_path = f"saved_models/run_meta_{name}.json"
                    load_path = last_path_a if os.path.exists(last_path_a) else (last_path_b if os.path.exists(last_path_b) else None)

                    # Before loading, check for metadata compatibility
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r') as f:
                                meta = json.load(f)
                            if meta.get('config_hash') != config_hash:
                                print(f"Warning: Config hash mismatch for {name}. Checkpoint may be incompatible. Skipping resume.")
                                continue
                        except Exception as e:
                            print(f"Warning: Could not read or validate meta file {meta_path}: {e}")

                    if load_path:
                        try:
                            ckpt = torch.load(load_path, map_location=device)
                            if isinstance(ckpt, dict) and 'model' in ckpt:
                                net.load_state_dict(ckpt['model'], strict=False)
                                try:
                                    optimizers[name].load_state_dict(ckpt.get('optimizer', {}))
                                except Exception:
                                    pass
                                print(f"Resumed {name.upper()} from {load_path}")
                            else:
                                if safe_load_checkpoint(net, load_path):
                                    print(f"Resumed {name.upper()} from {load_path}")
                        except Exception as e:
                            print(f"Warning: failed to resume {name}: {e}")
            for name, net in iter_items:
                print(f"\nTraining model: {name.upper()}")
                # Enable mixed precision scaler on CUDA if available
                scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
                loss = train_run(
                    net,
                    train_loader,
                    val_loader,
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
                    scaler=scaler,
                    max_batches=getattr(args, 'max_train_batches', None),
                    grad_accum_steps=max(1, getattr(args, 'grad_accum_steps', 1)),
                    holdout_loader=holdout_loader,
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
                # Record per-model validation stats for dynamic ensemble weighting
                try:
                    from train.train import evaluate as _eval
                    eval_ret = _eval(net, val_loader, criterion, aux_criterion, device, name, config.MULTI_TASK)
                    # Support evaluate returning 1,2, or 3 values
                    if isinstance(eval_ret, (list, tuple)):
                        v_loss = eval_ret[0] if len(eval_ret) >= 1 else float('inf')
                    else:
                        v_loss = float(eval_ret)
                    v_ppl = float(torch.exp(torch.tensor(v_loss)).item()) if (isinstance(v_loss, (int, float)) and v_loss > 0) else float('inf')
                    stats_path = 'ensemble_stats.json'
                    stats = {}
                    if os.path.exists(stats_path):
                        try:
                            with open(stats_path,'r',encoding='utf-8') as sf:
                                stats = json.load(sf) or {}
                        except Exception:
                            stats = {}
                    stats.setdefault('models', {})[name] = {
                        'val_loss': float(v_loss),
                        'val_ppl': v_ppl,
                        'step_count': optimizers[name].step_count
                    }
                    stats['config_hash'] = config_hash
                    with open(stats_path,'w',encoding='utf-8') as sf:
                        json.dump(stats, sf, indent=2)
                except Exception as e:
                    print(f"Warning: could not record ensemble stats for {name}: {e}")

                # Save run metadata for safe resume
                try:
                    meta_path = f"saved_models/run_meta_{name}.json"
                    run_meta = {
                        'config_hash': config_hash,
                        'step_count': optimizers[name].step_count,
                        'seed': user_seed,
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(run_meta, f, indent=4)
                except Exception as e:
                    print(f"Warning: Failed to save run metadata for {name}: {e}")


            # Load best model checkpoints after training round.
            for name, net in (iter_items if selected_model else models.items()):
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
            if config.USE_ACTIVE_LEARNING and not getattr(args, 'no_active_learning', False):
                # To select uncertain samples, we still need tensor inputs. We can get them from the loader.
                # This is less efficient than it could be, but keeps active learning functional.
                train_inputs_for_al = torch.cat([b[0] for b in train_loader], dim=0).to(device)

                # Select uncertain samples using configured strategy
                al_strategy = getattr(config, 'ACTIVE_LEARNING_STRATEGY', 'entropy')
                mc_passes = getattr(config, 'ACTIVE_LEARNING_MC_PASSES', 5)
                mc_enable = getattr(config, 'ACTIVE_LEARNING_ENABLE_MC_DROPOUT', True)
                uncertain_samples = active_learning.select_uncertain_samples(
                    models["lstm"],
                    train_inputs_for_al,
                    config.ACTIVE_LEARNING_TOP_K,
                    device,
                    strategy=al_strategy,
                    mc_passes=mc_passes,
                    enable_mc_dropout=mc_enable,
                )
                augmented_samples = augmentation.augment_data(uncertain_samples)

                # Normalize the augmented result to a single flat token list.
                def _pick_token_list(sample_any):
                    # If sample is a list of lists, take the first inner list.
                    if isinstance(sample_any, list) and sample_any and isinstance(sample_any[0], list):
                        return sample_any[0]
                    return sample_any

                chosen = _pick_token_list(augmented_samples[0]) if isinstance(augmented_samples, list) and augmented_samples else augmented_samples

                # If we already have numeric token ids, skip re-tokenization
                if isinstance(chosen, list) and chosen and isinstance(chosen[0], int):
                    numeric = chosen
                else:
                    # Otherwise assume string tokens and numericalize
                    numeric = data.tokenize_and_numericalize(chosen, word_to_index) if isinstance(chosen, list) else []

                # Build new sequences from the numeric list
                new_sequences = data.create_sequences(numeric, config.SEQUENCE_LENGTH) if numeric else []

                if new_sequences:
                    train_seqs.extend(new_sequences)
                    train_dataset = data.TextDataset(train_seqs)
                    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
                    print(f"Active learning added {len(new_sequences)} samples. New training set size: {len(train_dataset)}")
                    # Recompute class weights if enabled
                    try:
                        if getattr(config, 'CLASS_WEIGHTING', False) and not disable_class_weights:
                            class_weight = compute_class_weights(
                                train_dataset.sequences,
                                val_dataset.sequences,
                                vocab_size,
                                power=getattr(config, 'CLASS_WEIGHT_POWER', 0.5),
                                device=device
                            )
                            # Update criterion in-place if it supports .weight
                            if hasattr(criterion, 'weight') and class_weight is not None:
                                criterion.weight = class_weight
                            if writer is not None and class_weight is not None:
                                with torch.no_grad():
                                    step_tag = iteration + 1  # per AL round
                                    writer.add_scalar('ClassWeights/mean', float(class_weight.mean().item()), step_tag)
                                    writer.add_scalar('ClassWeights/std', float(class_weight.std().item()), step_tag)
                    except Exception as e:
                        print(f"Warning: failed to recompute class weights after AL: {e}")
                else:
                    print("Active learning found no valid augmented sequences; skipping augmentation update.")


    # Inference with Ensemble Voting
    ensemble_outputs = []
    token_mode = False  # default in case we don't run inference branch
    if mode in ('all', 'infer'):
        # Determine if a single model was requested for inference
        # Note: model selection is handled via args.model later when choosing runner model_choice
        # If user provided a prompt that is an identity question, answer deterministically
        if prompt_override is not None:
            lowered_prompt = str(prompt_override).lower()
            identity_prompts = [
                "who are you", "what is your name", "who made you", "who created you", "what are you"
            ]
            if any(q in lowered_prompt for q in identity_prompts):
                identity_answer = f"I am {config.BOT_NAME}, an AI assistant created by {config.BOT_CREATOR}."
                print("\nEnsemble Generated Text:\n", identity_answer)
                writer.close()
                # Respect skip_chat flag
                if not skip_chat and mode in ('all',):
                    chat_with_bot(models["lstm"], word_to_index, index_to_word, device,
                                  memory_size=0 if getattr(args, 'no_memory', False) else 5)
                else:
                    print("Prompt provided; skipping interactive chat and exiting.")
                return
        # If a prompt_override is provided, convert it using the same preprocessing
        # pipeline used during training/chat so the model receives token-aligned input.
        # Determine seed tokens: random from training data unless a prompt is provided and random-prompt is not requested
        use_random_seed = False
        try:
            use_random_seed = getattr(args, 'random_prompt', False) or (prompt_override is None)
        except NameError:
            use_random_seed = (prompt_override is None)

        if not use_random_seed and prompt_override is not None:
            # Use preprocessed prompt tokens joined back to a string so they map to vocab entries
            prompt_tokens = data.preprocess_data(str(prompt_override))
            start_words = " ".join(prompt_tokens)
            token_mode = False
        else:
            # Random seed: sample a random contiguous window from training tokens
            if len(numericalized_tokens) >= max(3, config.SEQUENCE_LENGTH):
                import random
                win = min(config.SEQUENCE_LENGTH, max(3, config.SEQUENCE_LENGTH // 2))
                start_idx = random.randint(0, max(0, len(numericalized_tokens) - win))
                start_words = numericalized_tokens[start_idx:start_idx + win]
            else:
                # Fallback to first few tokens
                start_words = numericalized_tokens[:max(3, min(len(numericalized_tokens), config.SEQUENCE_LENGTH))]
            token_mode = True

        try:
            selected_model = getattr(args, 'model', None)
        except NameError:
            selected_model = None
        # If not in token_mode (i.e., using string seeds) or when we build string context, prefix persona
        try:
            no_persona = getattr(args, 'no_persona', False)
        except NameError:
            no_persona = False
        # Retrieval augmentation: prepend similar sentences from JSONL when enabled
        try:
            if getattr(config, 'ENABLE_RETRIEVAL', False):
                from inference.retrieval import TfidfRetriever
                retr = TfidfRetriever(getattr(config, 'RETRIEVAL_INDEX_PATH', 'data/data.jsonl'),
                                      min_chars=int(getattr(config, 'RETRIEVAL_MIN_CHARS', 20)))
                k = int(getattr(config, 'RETRIEVAL_TOP_K', 3))
                if isinstance(start_words, str) and start_words:
                    hits = retr.top_k(start_words, k=k)
                    thr = float(getattr(config, 'RETRIEVAL_SCORE_THRESHOLD', 0.0))
                    # Filter by threshold and keep non-empty
                    hits = [s for (score, s) in hits if score >= thr and s]
                    if hits:
                        prefix = str(getattr(config, 'RECALL_PREFIX', 'Context: '))
                        start_words = (" ".join([prefix + h for h in hits]) + " \n " + start_words)
        except Exception:
            pass

        # Use ActiveRunner for inference to restore vocab directly from checkpoints
        runner = get_runner()
        # Build a prompt string for the runner
        if isinstance(start_words, str):
            prompt_text = start_words
        else:
            # Convert numeric tokens to a space-separated string via current mapping
            prompt_text = " ".join([index_to_word.get(t, "<UNK>") for t in start_words])
        if not no_persona and isinstance(prompt_text, str) and prompt_text:
            prompt_text = (config.PERSONA_PREFIX + " " + prompt_text) if getattr(config, 'PERSONA_PREFIX', '') else prompt_text

        # Choose model: explicit CLI selection, else optional ensemble, else auto
        if selected_model in ('rnn','gru','lstm','transformer'):
            model_choice = selected_model
        else:
            model_choice = 'ensemble' if bool(getattr(config, 'USE_ENSEMBLE', False)) else 'auto'

        output_text = runner.run(
            prompt_text,
            config.NUM_WORDS_TO_GENERATE,
            config.TEMPERATURE,
            getattr(config, 'DECODING', 'sampling'),
            getattr(config, 'TOP_K', 0),
            getattr(config, 'TOP_P', 1.0),
            model_choice=model_choice
        )
        try:
            if not getattr(args, 'no_refine', False):
                output_text = refine_text(output_text)
        except NameError:
            output_text = refine_text(output_text)
        ensemble_outputs = [(output_text, 1)]

    if mode in ('all', 'infer'):
        final_output = utils.weighted_ensemble_output(ensemble_outputs)
        try:
            if not getattr(args, 'no_refine', False):
                final_output = refine_text(final_output)
        except NameError:
            final_output = refine_text(final_output)

        # Adaptive Temperature Adjustment using PID (via ActiveRunner)
        temperature = config.TEMPERATURE
        for _ in range(3):
            # Use the same preprocessed start words if a prompt_override is provided
            if prompt_override is not None:
                temp_prompt_tokens = data.preprocess_data(prompt_override)
                temp_start = " ".join(temp_prompt_tokens)
            else:
                temp_start = config.START_WORDS if isinstance(config.START_WORDS, str) else " ".join([index_to_word.get(t, "<UNK>") for t in (config.START_WORDS or [])])

            generated = runner.run(
                temp_start,
                config.NUM_WORDS_TO_GENERATE,
                temperature,
                getattr(config, 'DECODING', 'sampling'),
                getattr(config, 'TOP_K', 0),
                getattr(config, 'TOP_P', 1.0),
                model_choice='lstm' if selected_model == 'lstm' else model_choice
            )
            try:
                if not getattr(args, 'no_refine', False):
                    generated = refine_text(generated)
            except NameError:
                generated = refine_text(generated)
            temperature += pid_temp.update(len(generated.split()) / config.NUM_WORDS_TO_GENERATE)
            temperature = max(0.5, min(1.5, temperature))
            print(f"\nGenerated Text at Temperature {temperature}:\n{generated}")

        print("\nEnsemble Generated Text:\n", final_output)

        # Structured inference logging (JSONL)
        try:
            import hashlib
            import time
            prompt_src = str(prompt_override) if prompt_override is not None else str(start_words)
            phash = hashlib.sha256(prompt_src.encode('utf-8')).hexdigest()[:16]
            log_rec = {
                'ts': time.time(),
                'prompt_hash': phash,
                'prompt_preview': prompt_src[:80],
                'tokens_generated': len(final_output.split()),
                'ensemble_mode': getattr(config, 'ENSEMBLE_MODE', None),
                'trace_file': getattr(config, 'TRACE_FILENAME', None) if getattr(config, 'GENERATION_TRACE', False) else None
            }
            # If trace file exists and is JSON, attempt to extract summary
            tfile = getattr(config, 'TRACE_FILENAME', None)
            if tfile and os.path.exists(tfile):
                try:
                    with open(tfile,'r',encoding='utf-8') as tf:
                        trace_data = json.load(tf)
                    if isinstance(trace_data, dict) and 'summary' in trace_data:
                        log_rec['metrics'] = trace_data['summary']
                    elif isinstance(trace_data, list):
                        # legacy list trace summary last element
                        cand = trace_data[-1]
                        if isinstance(cand, dict) and 'summary' in cand:
                            log_rec['metrics'] = cand['summary']
                except Exception:
                    pass
            os.makedirs('logs', exist_ok=True)
            with open('logs/inference.log','a',encoding='utf-8') as lf:
                lf.write(json.dumps(log_rec) + '\n')
        except Exception:
            pass

        writer.close()

    # Start the interactive chat session after building/loading models.
    # If prompt_override was provided, we still allow entering chat when --chat-only
    # was requested; otherwise, respect skip_chat and mode.
    if mode in ('all','infer') and token_mode:
        # If the user provided a prompt and didn't request chat-only, exit after generation
        if not getattr(args, 'chat_only', False):
            print("Prompt provided; skipping interactive chat and exiting.")
            return

    # If chat-only was requested, always enter chat (after attempting to load checkpoints)
    if mode in ('all',) and getattr(args, 'chat_only', False):
        # Attempt to load checkpoints if present (using safe loader)
        for name, net in models.items():
            checkpoint_path = f"saved_models/best_model_{name}.pth"
            if os.path.exists(checkpoint_path):
                try:
                    safe_load_checkpoint(net, checkpoint_path)
                except Exception as e:
                    print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        # Enter interactive chat
        chat_with_bot(models["lstm"], word_to_index, index_to_word, device,
                      memory_size=0 if getattr(args, 'no_memory', False) else config.CHAT_MEMORY_SIZE)
        return

    # Honor skip_chat flag for non-interactive runs.
    if mode in ('all',) and not skip_chat:
        chat_with_bot(models["lstm"], word_to_index, index_to_word, device,
                      memory_size=0 if getattr(args, 'no_memory', False) else config.CHAT_MEMORY_SIZE)
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
    parser.add_argument('--random-prompt', action='store_true', help='Use a random prompt seed from training data (ignores --prompt)')
    parser.add_argument('--no-refine', action='store_true', help='Disable text refinement post-processing of generated text')
    parser.add_argument('--no-memory', action='store_true', help='Do not use chat memory context during generation/chat')
    parser.add_argument('--model', choices=['rnn','gru','lstm','transformer','mlp'], help='Select a single model to generate with instead of ensemble')
    parser.add_argument('--no-active-learning', action='store_true', help='Disable active learning augmentation during training')
    parser.add_argument('--batch-size', type=int, help='Override training batch size')
    parser.add_argument('--max-train-batches', type=int, help='Limit number of training batches per epoch for quick smoke tests')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto', help='Select compute device (auto/cpu/cuda)')
    parser.add_argument('--no-persona', action='store_true', help='Disable persona prefixing in chat/inference context')
    # Chat UX controls
    parser.add_argument('--stream', action='store_true', help='Stream chat responses token-by-token')
    parser.add_argument('--style', choices=['friendly','concise','formal'], help='Chat response style')
    parser.add_argument('--memory-size', type=int, help='Override chat rolling memory size')
    parser.add_argument('--memory-mode', choices=['session','full','summary'], help='Select memory mode')
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='Accumulate gradients over N steps before optimizer step')
    parser.add_argument('--decoding', choices=['sampling','beam'], help='Decoding strategy: sampling (default) or beam')
    parser.add_argument('--beam-size', type=int, help='Beam width when using beam search decoding')
    parser.add_argument('--length-penalty', type=float, help='Length penalty for beam search (>=0)')
    parser.add_argument('--early-stop', action='store_true', help='Enable early stop on EOS during generation')
    parser.add_argument('--no-early-stop', action='store_true', help='Disable early stop on EOS during generation')
    parser.add_argument('--min-length', type=int, help='Minimum tokens before allowing EOS')
    parser.add_argument('--eos-token', type=str, help='Token string to treat as EOS (must be in vocabulary)')
    parser.add_argument('--trace', action='store_true', help='Enable generation tracing regardless of config')
    parser.add_argument('--no-trace', action='store_true', help='Disable generation tracing regardless of config')
    parser.add_argument('--trace-file', type=str, help='Override trace output filename')
    parser.add_argument('--ensemble-mode', choices=['simple','learned','learned_dynamic'], help='Override ensemble mode at runtime')
    parser.add_argument('--seed', type=int, help='Global RNG seed for reproducibility')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoints if available')
    parser.add_argument('--warmup-steps', type=int, help='Linear warmup steps before scheduler starts')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms (slower)')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable inverse-frequency class weighting')
    parser.add_argument('--model-scale', choices=['tiny','small','base','medium','large','xl'], help='Scale model width/depth presets at runtime')
    parser.add_argument('--use-output-attention', action='store_true', help='Enable LSTM output self-attention block for better sequence modeling')
    # Data pipeline utilities
    parser.add_argument('--compile-data', action='store_true', help='Compile raw text into data.txt and data.jsonl, then exit')
    parser.add_argument('--data-report', action='store_true', help='Print dataset summary from JSONL/meta, then exit')
    args = parser.parse_args()

    # Runtime overrides for tracing
    if getattr(args, 'trace', False):
        try:
            config.GENERATION_TRACE = True
        except Exception:
            pass
    if getattr(args, 'no_trace', False):
        try:
            config.GENERATION_TRACE = False
        except Exception:
            pass
    if getattr(args, 'trace_file', None):
        try:
            config.TRACE_FILENAME = args.trace_file
        except Exception:
            pass
    if getattr(args, 'ensemble_mode', None):
        try:
            config.ENSEMBLE_MODE = args.ensemble_mode
        except Exception:
            pass
    # Runtime override for model scale
    if getattr(args, 'model_scale', None):
        try:
            config.MODEL_SCALE = args.model_scale
        except Exception:
            pass
    if getattr(args, 'use_output_attention', False):
        try:
            config.USE_OUTPUT_ATTENTION = True
        except Exception:
            pass

    # Handle data utilities and exit early if requested
    if getattr(args, 'compile_data', False) or getattr(args, 'data_report', False):
        # Ensure directories exist
        try:
            data.setup_directories()
        except Exception:
            pass
        if getattr(args, 'compile_data', False):
            print("Compiling data into data.txt and data.jsonl ...")
            try:
                data.compile_data()
                print("Data compilation complete.")
            except Exception as e:
                print(f"Data compilation failed: {e}")
        if getattr(args, 'data_report', False):
            print("Generating dataset report ...")
            try:
                import json
                import os
                meta_path = getattr(config, 'JSONL_META_PATH', 'data/data_meta.json')
                jsonl_path = getattr(config, 'JSONL_PATH', 'data/data.jsonl')
                # If JSONL missing or empty, attempt fallback build from data.txt
                if (not os.path.exists(jsonl_path)) or (os.path.getsize(jsonl_path) == 0):
                    print("JSONL missing/empty — building from data.txt fallback ...")
                    try:
                        data.build_jsonl_from_data_file(jsonl_path)
                        # After building, try to run validation/meta by invoking compile_data's tail logic
                        # Simulate by reusing the validator/meta paths if desired
                        # We'll do a light validation here
                        total_lines = sum(1 for _ in open(jsonl_path, 'r', encoding='utf-8')) if os.path.exists(jsonl_path) else 0
                        print(f"Built JSONL with {total_lines} lines")
                    except Exception as e:
                        print(f"Fallback JSONL build failed: {e}")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        meta = json.load(mf)
                    print("Dataset Summary:")
                    print(f"  Total lines: {meta.get('total_lines')}\n  Avg length: {meta.get('avg_length'):.2f}\n  Avg alpha ratio: {meta.get('avg_alpha_ratio'):.3f}")
                    per_source = meta.get('per_source', {})
                    if per_source:
                        print("  Per-source counts (top 10):")
                        for k, v in sorted(per_source.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                            print(f"    - {k}: {v}")
                elif os.path.exists(jsonl_path):
                    # Quick on-the-fly stats if meta missing
                    total = 0
                    total_len = 0
                    total_alpha = 0.0
                    per_source = {}
                    def _alpha_ratio_loc(s):
                        letters = sum(ch.isalpha() for ch in s)
                        return letters / max(1, len(s))
                    with open(jsonl_path, 'r', encoding='utf-8') as jf:
                        for line in jf:
                            try:
                                rec = json.loads(line)
                                s = rec.get('text', '')
                                src = rec.get('source', 'unknown')
                                total += 1
                                total_len += len(s)
                                total_alpha += _alpha_ratio_loc(s)
                                per_source[src] = per_source.get(src, 0) + 1
                            except Exception:
                                pass
                    if total > 0:
                        print("Dataset Summary (on-the-fly):")
                        print(f"  Total lines: {total}\n  Avg length: {total_len/max(1,total):.2f}\n  Avg alpha ratio: {total_alpha/max(1,total):.3f}")
                        print("  Per-source counts (top 10):")
                        for k, v in sorted(per_source.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                            print(f"    - {k}: {v}")
                    else:
                        print("No JSONL records found to report.")
                else:
                    print("No dataset files found. Consider running with --compile-data first.")
            except Exception as e:
                print(f"Dataset report failed: {e}")
        # Exit after utility action(s)
        import sys
        sys.exit(0)

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
                main(mode='infer', skip_chat=False)
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
            # Apply CLI overrides to config for decoding-related options
            if args.batch_size is not None and args.batch_size > 0:
                config.BATCH_SIZE = args.batch_size
            if args.decoding is not None:
                config.DECODING = args.decoding
            if args.beam_size is not None:
                config.BEAM_SIZE = max(1, args.beam_size)
            if args.length_penalty is not None:
                config.LENGTH_PENALTY = max(0.0, args.length_penalty)
            if args.min_length is not None:
                config.MIN_LENGTH = max(0, args.min_length)
            if args.early_stop:
                config.EARLY_STOP = True
            if args.no_early_stop:
                config.EARLY_STOP = False
            if args.eos_token is not None:
                config.EOS_TOKEN = args.eos_token

            main(mode=args.mode, skip_chat=args.skip_chat, epochs_override=args.epochs, prompt_override=args.prompt, scrape=args.scrape)

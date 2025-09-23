import os
 


def run_main_with_args(args_list):
    # Run main as a script so the argparse __main__ block executes.
    import sys
    import runpy
    sys_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + args_list
    try:
        runpy.run_module('main', run_name='__main__')
    finally:
        sys.argv = sys_argv


def test_scrape_flag_triggers_pipeline(tmp_path, monkeypatch):
    called = {'wiki': False, 'live': False, 'compile': False}

    def fake_setup_directories():
        os.makedirs(tmp_path / "data", exist_ok=True)

    def fake_fetch_wikipedia_articles(topics):
        called['wiki'] = True

    def fake_fetch_live_data(urls):
        called['live'] = True

    def fake_compile_data():
        called['compile'] = True

    # Patch data functions
    import data.data as data_module
    monkeypatch.setattr(data_module, 'setup_directories', fake_setup_directories)
    monkeypatch.setattr(data_module, 'fetch_wikipedia_articles', fake_fetch_wikipedia_articles)
    monkeypatch.setattr(data_module, 'fetch_live_data', fake_fetch_live_data)
    monkeypatch.setattr(data_module, 'compile_data', fake_compile_data)

    # Stub model classes so main can instantiate models cheaply
    import model as model_module

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return

        def init_hidden(self, batch_size=1, device=None):
            return None

        def __call__(self, *args, **kwargs):
            # Return a tensor of logits for a vocabulary of size 2
            import torch
            batch_size = 1
            vocab_size = 3
            logits = torch.randn(batch_size, vocab_size)
            # If called with (input_tensor, hidden) return (logits, aux, hidden)
            if len(args) >= 2:
                hidden = args[1]
                return logits, None, hidden

            # If called with a single tensor, inspect dtype to decide return shape:
            if len(args) == 1 and hasattr(args[0], 'dtype'):
                tensor = args[0]
                # Transformer path expects (logits, aux)
                if tensor.dtype == torch.long:
                    return logits, None
                # MLP path expects logits only (float inputs)
                return logits

            # Default: logits only
            return logits

    monkeypatch.setattr(model_module, 'RNNTextGenerationModel', DummyModel, raising=False)
    monkeypatch.setattr(model_module, 'GRUTextGenerationModel', DummyModel, raising=False)
    monkeypatch.setattr(model_module, 'LSTMTextGenerationModel', DummyModel, raising=False)
    monkeypatch.setattr(model_module, 'TransformerTextGenerationModel', DummyModel, raising=False)
    monkeypatch.setattr(model_module, 'MLPModel', DummyModel, raising=False)

    # Stub MetaOptimizer used in main
    import train.optimizer as opt_module

    class DummyMetaOpt:
        def __init__(self, model, base_lr=None, scheduler_step=None, gamma=None):
            pass

        def step(self, loss):
            pass

        def get_lr(self):
            return 0.0

    monkeypatch.setattr(opt_module, 'MetaOptimizer', DummyMetaOpt, raising=False)

    # Stub SummaryWriter to avoid tensorboard dependency
    try:
        import torch.utils.tensorboard as tb
        class DummyWriter:
            def __init__(self, *args, **kwargs):
                pass
            def close(self):
                pass
        monkeypatch.setattr(tb, 'SummaryWriter', DummyWriter, raising=False)
    except Exception:
        pass

    # Run main with --scrape in infer mode and skip chat; it should call the patched functions
    run_main_with_args(['--mode', 'infer', '--skip-chat', '--scrape'])

    assert called['wiki'] and called['live'] and called['compile']

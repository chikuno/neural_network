# verification script removed - file left as a marker
import importlib
packages = ['torch','torchvision','torchaudio','tensorboard','nltk','deep_translator','bs4','requests','numpy','tqdm','matplotlib']
for p in packages:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, '__version__', str(type(m)))
        print(p, 'OK', ver)
    except Exception as e:
        print(p, 'ERROR', e)

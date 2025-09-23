import importlib
import os
import sys

# Ensure repo root is on sys.path so local packages can be imported when run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

modules = ['data.data','data.augmentation','train.train','inference.generate','main']

for m in modules:
    try:
        importlib.import_module(m)
        print(m, 'import OK')
    except Exception as e:
        print(m, 'import ERROR:', type(e).__name__, e)

import importlib
import traceback

modules = ['model.pid_controller', 'train.optimizer', 'main']
for m in modules:
    try:
        importlib.import_module(m)
        print(m + ' import OK')
    except Exception:
        print(m + ' import ERROR:')
        traceback.print_exc()

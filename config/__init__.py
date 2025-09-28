"""Package initializer for config.

Exports a single attribute `config` so existing code using
    from config import config
continues to work. The underlying `config/config.py` defines a set of
UPPER_CASE constants. We wrap those in a lightweight mutable namespace.

If new constants are added to `config/config.py`, they will automatically
be included upon first import. Mutations to attributes on `config` are
local (they won't write back to the module variables) but that matches
how the project already performs runtime overrides (attribute assignment
on the imported object).
"""

from types import SimpleNamespace
import importlib

_raw_module = importlib.import_module('.config', __name__)

def _build_namespace(mod):
    data = {}
    for k, v in vars(mod).items():
        if k.isupper():  # export only constants / settings
            data[k] = v
    return SimpleNamespace(**data)

config = _build_namespace(_raw_module)

__all__ = ['config']

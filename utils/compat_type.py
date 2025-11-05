import numpy as np

# Compact compatibility for removed numpy alias names (np.int, np.float, np.bool, ...)
_alias_fallbacks = {'int': np.int32, 'float': float, 'float32': float, 'float64': float, 'bool': bool, 'object': object}

def np_alias(name: str):
    """Return numpy.alias if present, otherwise a safe Python fallback."""
    return getattr(np, name, _alias_fallbacks.get(name, object))

# common ready-to-use aliases
np_int   = np_alias('int')

__all__ = ['np_alias', 'np_int']
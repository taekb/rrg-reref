import importlib

__all__ = ['maira2', 'medgemma', 'medversa', 'chexagent', 'llava_rad', 'chex_mimic', 'libra']

def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

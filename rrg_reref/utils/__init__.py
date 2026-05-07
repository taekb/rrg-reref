import importlib

__all__ = ['constants', 'models', 'data', 'logging', 'green', 'crimson', 'llm_api']

def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

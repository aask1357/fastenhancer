import importlib


def get_wrapper(type):
    module = importlib.import_module(f"wrappers.{type}")
    Wrapper = module.ModelWrapper
    return Wrapper

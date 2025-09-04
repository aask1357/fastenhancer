import importlib
from .modelwrapper import AudioModelWrapper


def get_wrapper(model : str) -> AudioModelWrapper:
    module = importlib.import_module(f"models.{model}")
    Wrapper = module.ModelWrapper
    return Wrapper

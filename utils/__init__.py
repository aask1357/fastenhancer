from .hparams import get_hparams, HParams
from .summarize import summarize, plot_param_and_grad, plot_numpy
from .grad_clip import clip_grad_norm_local
from .verbose import verbose
from .measure_metrics import Metrics


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter
        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False
    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

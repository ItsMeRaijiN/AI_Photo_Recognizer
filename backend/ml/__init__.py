from __future__ import annotations

from .config import Config, parse_args

__all__ = ["Config", "parse_args", "train", "run_optuna", "evaluate_main"]


def train(*args, **kwargs):
    from .train import train as _train
    return _train(*args, **kwargs)


def run_optuna(*args, **kwargs):
    from .train import run_optuna as _run_optuna
    return _run_optuna(*args, **kwargs)


def evaluate_main(*args, **kwargs):
    from .evaluate import main as _main
    return _main(*args, **kwargs)

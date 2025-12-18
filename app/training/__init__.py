"""Training and evaluation module."""

from .trainer import Trainer, EarlyStopping
from .evaluation import analyze_generalization, evaluate_model

__all__ = ["Trainer", "EarlyStopping", "analyze_generalization", "evaluate_model"]

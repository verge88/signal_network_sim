"""Просмотр, визуализация и экспорт результатов экспериментов."""

from .model import ExperimentResult, ChartSpec
from .loader import load_experiment

__all__ = ["ExperimentResult", "ChartSpec", "load_experiment"]

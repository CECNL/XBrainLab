from .confusion_matrix import ConfusionMatrixWindow
from .evaluation_table import EvaluationTableWindow
from .model_output import ModelOutputWindow

EVALUATION_MODULE_LIST = [
    ConfusionMatrixWindow, EvaluationTableWindow, ModelOutputWindow
]

__all__ = [
    'EVALUATION_MODULE_LIST',
    'ConfusionMatrixWindow',
    'EvaluationTableWindow',
    'ModelOutputWindow'
]

from enum import Enum


class Metric(Enum):
    """Utility class for evaluation metric."""
    ACC = 'Accuracy (%)'
    AUC = 'Area under ROC-curve'
    KAPPA = 'kappa value'

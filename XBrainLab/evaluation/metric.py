from enum import Enum

class Metric(Enum):
    ACC = 'Accuracy (%)'
    AUC = 'Area under ROC-curve'
    KAPPA = 'kappa value'

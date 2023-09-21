from enum import Enum


class SplitUnit(Enum):
    """Utility class for dataset splitting unit."""
    RATIO = 'Ratio'
    NUMBER = 'Number'
    MANUAL = 'Manual'
    KFOLD = 'K Fold'

class TrainingType(Enum):
    """Utility class for model training type."""
    FULL = 'Full Data'
    IND = 'Individual'

class SplitByType(Enum):
    """Utility class for dataset splitting type."""
    DISABLE = 'Disable'
    SESSION = 'By Session'
    SESSION_IND = 'By Session (Independent)'
    TRIAL = 'By Trial'
    TRIAL_IND = 'By Trial (Independent)'
    SUBJECT = 'By Subject'
    SUBJECT_IND = 'By Subject (Independent)'

class ValSplitByType(Enum):
    """Utility class for dataset splitting type for validation set."""
    DISABLE = 'Disable'
    SESSION = 'By Session'
    TRIAL = 'By Trial'
    SUBJECT = 'By Subject'

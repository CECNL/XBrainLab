from enum import Enum

class SplitUnit(Enum):
    RATIO = 'Ratio'
    NUMBER = 'Number'
    MANUAL = 'Manual'
    KFOLD = 'K Fold'
    
class TrainingType(Enum):
    FULL = 'Full Data'
    IND = 'Individual'

class SplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    SESSION_IND = 'By Session (Independent)'
    TRIAL = 'By Trial'
    TRIAL_IND = 'By Trial (Independent)'
    SUBJECT = 'By Subject'
    SUBJECT_IND = 'By Subject (Independent)'

class ValSplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    TRIAL = 'By Trial'
    SUBJECT = 'By Subject'
   
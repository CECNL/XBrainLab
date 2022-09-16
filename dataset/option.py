from enum import Enum

class SplitUnit(Enum):
    RATIO = 'Ratio'
    NUMBER = 'Number'
    KFOLD = 'K Fold'
    
class TrainingType(Enum):
    FULL = 'Full Data'
    IND = 'Individual'

class SplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    SESSION_IND = 'By Session (Independent)'
    TRAIL = 'By Trail'
    TRAIL_IND = 'By Trail (Independent)'
    SUBJECT = 'By Subject'
    SUBJECT_IND = 'By Subject (Independent)'

class ValSplitByType(Enum):
    DISABLE = 'Disable'
    SESSION = 'By Session'
    TRAIL = 'By Trail'
    SUBJECT = 'By Subject'
   
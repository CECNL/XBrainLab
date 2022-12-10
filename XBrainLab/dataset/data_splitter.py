from .option import SplitUnit, SplitByType, ValSplitByType
from ..utils import validate_type

class DataSplitter():
    def __init__(self, split_type, value_var=None, split_unit=None, is_option=True):
        validate_type(split_type, (SplitByType, ValSplitByType) ,"split_type")
        if split_unit:
            validate_type(split_unit, SplitUnit ,"split_unit")
        self.is_option = is_option
        self.split_type = split_type
        self.text = text = split_type.value
        self.value_var = value_var
        self.split_unit = split_unit
    
    def is_valid(self):
        if self.value_var is None:
            return False
        if self.split_unit is None:
            return False
        
        
        if self.split_unit == SplitUnit.RATIO:
            try:
                val = float(self.value_var)
                if 0 <= val <= 1:
                    return True
            except ValueError:
                return False   
        elif self.split_unit == SplitUnit.NUMBER:
            return str(self.value_var).isdigit()
        elif self.split_unit == SplitUnit.KFOLD:
            val = str(self.value_var)
            if val.isdigit():
                return 0 < int(val)
        return False

    # getter
    def get_value(self):
        return float(self.value_var)
    
    def get_raw_value(self):
        return self.value_var
        
    def get_split_unit(self):
        return self.split_unit
    
    def get_split_unit_repr(self):
        return f"{self.split_unit.__class__.__name__}.{self.split_unit.name}"

    def get_split_type_repr(self):
        return f"{self.split_type.__class__.__name__}.{self.split_type.name}"

    
class DataSplittingConfig():
    def __init__(self, train_type, is_cross_validation, val_splitter_list, test_splitter_list):
        self.train_type = train_type # TrainingType
        self.is_cross_validation = is_cross_validation
        self.val_splitter_list = val_splitter_list
        self.test_splitter_list = test_splitter_list

    def get_splitter_option(self):
        return self.val_splitter_list, self.test_splitter_list

    def get_train_type_repr(self):
        return f"{self.train_type.__class__.__name__}.{self.train_type.name}"


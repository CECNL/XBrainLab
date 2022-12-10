from ..load_data import Raw
from ..utils import validate_list_type
from copy import deepcopy
import tkinter as tk

class PreprocessBase:
    def __init__(self, preprocessed_data_list):
        self.preprocessed_data_list = deepcopy(preprocessed_data_list)
        self.check_data()

    def check_data(self):
        if not self.preprocessed_data_list:
            raise ValueError("No valid data is loaded")
        validate_list_type(self.preprocessed_data_list, Raw, 'preprocessed_data_list')
        
    def get_preprocessed_data_list(self):
        return self.preprocessed_data_list

    def get_preprocess_desc(self, *args, **kargs):
        raise NotImplementedError
    
    def data_preprocess(self, *args, **kargs):
        for preprocessed_data in self.preprocessed_data_list:
            self._data_preprocess(preprocessed_data, *args, **kargs)
            preprocessed_data.add_preprocess(self.get_preprocess_desc(*args, **kargs))
        return self.preprocessed_data_list

    def _data_preprocess(self, *args, **kargs):
        raise NotImplementedError
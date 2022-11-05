from ..base import TopWindow, InitWindowValidateException
from ..dataset.data_holder import Raw
from copy import deepcopy
import tkinter as tk

class PreprocessBase(TopWindow):
    def __init__(self, parent, title, preprocessed_data_list):
        super().__init__(parent, title)
        self.preprocessed_data_list = deepcopy(preprocessed_data_list)
        self.return_data = None
        self.check_data()

    def check_data(self):
        if not self.preprocessed_data_list:
            raise InitWindowValidateException(window=self, message="No valid data is loaded")
        for preprocessed_data in self.preprocessed_data_list:
            if type(preprocessed_data) != Raw:
                raise InitWindowValidateException(window=self, message=f"Invalid data type, got ({type(preprocessed_data)})")

    def _get_result(self):
        return self.return_data

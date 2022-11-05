from .base import LoadBase
from .info import RawInfo
import tkinter as tk
from .array_setter import ArrayInfoSetter

class LoadArray(LoadBase):
    def __init__(self, parent, title):
        self.raw_info = RawInfo()
        super().__init__(parent, title)

    def reset(self):
        super().reset()
        self.raw_info.reset()

    def handle_array(self, filepath, selected_data):
        if not self.raw_info.is_info_complete():
            array_info = ArrayInfoSetter(self, filepath, selected_data, self.raw_info, type_ctrl=self.type_ctrl.get()).get_result()
            if not array_info:
                return False
            self.array_info = array_info

        data = self.array_info.generate_mne(filepath, selected_data, self.type_ctrl.get())
        return data
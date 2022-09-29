from .array import LoadArray
from .info import DictInfo
import tkinter as tk
from .dict_setter import DictInfoSetter

class LoadDict(LoadArray):
    def __init__(self, parent, title):
        self.dict_info = DictInfo()
        super().__init__(parent, title)
        # ==== status table ====
        self.clear_btn = tk.Button(self.stat_frame, text="Clear selected keys", command=self._clear_key)
        self.clear_btn.config(state='disabled')
        self.clear_btn.grid(row=5, column=0)

    def reset(self):
        super().reset()
        self.dict_info.reset()

    def _clear_key(self):
        self.dict_info.reset_keys()
        self.clear_btn.config(state='disabled')

    def handle_dict(self, filepath, selected_data):
        if not self.dict_info.is_info_complete(selected_data):
            dict_info = DictInfoSetter(self, filepath, selected_data, self.dict_info, type_ctrl=self.type_ctrl.get()).get_result()
            if not dict_info:
                return False
            self.dict_info = dict_info
            self.clear_btn.config(state='active')

        mne_data = self.dict_info.generate_mne(filepath, selected_data, self.type_ctrl.get())
        return mne_data
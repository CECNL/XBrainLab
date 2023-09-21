import tkinter as tk

from .array import LoadArray
from .dict_setter import DictInfoSetter
from .info import DictInfo


class LoadDict(LoadArray):
    def __init__(self, parent, title):
        super().__init__(parent, title)
        # ==== status table ====
        self.clear_btn = tk.Button(
            self.stat_frame, text="Clear selected keys", command=self._clear_key
        )
        self.clear_btn.config(state='disabled')
        self.clear_btn.grid(row=self.stat_frame_row_count, column=0)

    def reset(self):
        self.dict_info = DictInfo()
        super().reset()

    def _clear_key(self):
        self.dict_info.reset_keys()
        self.clear_btn.config(state='disabled')

    def handle_dict(self, filepath, selected_data):
        if not self.dict_info.is_info_complete(selected_data):
            dict_info_module = DictInfoSetter(
                self,
                filepath, selected_data, self.dict_info,
                type_ctrl=self.type_ctrl.get()
            )
            dict_info = dict_info_module.get_result()

            if not dict_info:
                return False
            self.dict_info = dict_info
            self.clear_btn.config(state='active')

        mne_data, generation_script = self.dict_info.generate_mne(
            filepath, selected_data, self.type_ctrl.get()
        )
        self.script_history += generation_script
        return mne_data

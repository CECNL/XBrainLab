from .base import PanelBase
import tkinter as tk

class PreprocessPanel(PanelBase):
    def __init__(self, parent, **args):
        super().__init__(parent, text='Preprocess history', **args)
        self.history_list = tk.Listbox(self)
        self.history_list.pack(fill=tk.BOTH, expand=True)
            
    def update_panel(self, preprocessed_data_list):
        while self.history_list.size() > 0:
            self.history_list.delete(0)
        if not preprocessed_data_list:
            return
        preprocessed_data = preprocessed_data_list[0]
        if preprocessed_data.get_preprocess_history():
            self.history_list.insert(tk.END, *preprocessed_data.get_preprocess_history())
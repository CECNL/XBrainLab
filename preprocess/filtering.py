from .base import PreprocessBase
from ..base import ValidateException
import tkinter as tk

class Filtering(PreprocessBase):
    command_label = "Filtering"
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Filtering", preprocessed_data_list)
        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}
        
        self.rowconfigure([0,1], weight=1)
        self.columnconfigure([0,1], weight=1)

        tk.Label(self, text="Lower pass-band edge: ").grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(self, textvariable=self.field_var['l_freq']).grid(row=0, column=1, padx=5)
        tk.Label(self, text="Upper pass-band edge: ").grid(row=1, column=0, padx=5, pady=10)
        tk.Entry(self, textvariable=self.field_var['h_freq']).grid(row=1, column=1, padx=5)
        tk.Button(self, text="Confirm", command=self._data_preprocess, width=8).grid(row=2, columnspan=2)

    def get_preprocess_desc(self, l_freq, h_freq):
        return f"Filtering {l_freq} ~ {h_freq}"

    def _data_preprocess(self):
        # Check if input is empty
        if not self.field_var['l_freq'].get() and not self.field_var['h_freq'].get():
            raise ValidateException(window=self, message="No Input")

        l_freq = float(self.field_var['l_freq'].get()) if self.field_var['l_freq'].get().strip() else None
        h_freq = float(self.field_var['h_freq'].get()) if self.field_var['h_freq'].get().strip() else None

        for preprocessed_data in self.preprocessed_data_list:
            preprocessed_data.get_mne().load_data()
            preprocessed_data.get_mne().filter(l_freq=l_freq, h_freq=h_freq)
            preprocessed_data.add_preprocess(self.get_preprocess_desc(l_freq, h_freq))

        self.return_data = self.preprocessed_data_list
        self.destroy()

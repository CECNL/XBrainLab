import tkinter as tk

from ..base import ValidateException
from .base import PreprocessBase

from XBrainLab import preprocessor as Preprocessor

class Resample(PreprocessBase):
    command_label = "Resample"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.Resample(preprocessed_data_list)
        super().__init__(parent, "Resample", preprocessor)
        data_field = ["sfreq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        self.rowconfigure([0,1], weight=1)
        self.columnconfigure([0,1], weight=1)

        tk.Label(self, text="Sampling Rate: ").grid(row=0, column=0, pady=10, padx=5)
        tk.Entry(self, textvariable=self.field_var['sfreq']).grid(row=0, column=1, pady=10, padx=5)
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=1, columnspan=2)

    def _data_preprocess(self):
        # Check Input is Valid
        if self.field_var['sfreq'].get().strip() == "":
            raise ValidateException(window=self, message="No Input")

        try:
            sfreq = float(self.field_var['sfreq'].get())
            self.return_data = self.preprocessor.data_preprocess(sfreq)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        self.script_history.add_cmd(f'sfreq={repr(sfreq)}')
        self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.Resample, sfreq=sfreq)')
        self.ret_script_history = self.script_history

        self.destroy()
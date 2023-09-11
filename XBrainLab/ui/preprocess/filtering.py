import tkinter as tk

from ..base import ValidateException
from .base import PreprocessBase

from XBrainLab import preprocessor as Preprocessor

class Filtering(PreprocessBase):
    command_label = "Filtering"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.Filtering(preprocessed_data_list)
        super().__init__(parent, "Filtering", preprocessor)
        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}
        
        self.rowconfigure([0,1], weight=1)
        self.columnconfigure([0,1], weight=1)

        tk.Label(self, text="Lower pass-band edge: ").grid(
            row=0, column=0, padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.field_var['l_freq']).grid(
            row=0, column=1, padx=5
        )
        tk.Label(self, text="Upper pass-band edge: ").grid(
            row=1, column=0, padx=5, pady=10
        )
        tk.Entry(self, textvariable=self.field_var['h_freq']).grid(
            row=1, column=1, padx=5
        )
        tk.Button(self, text="Confirm", command=self._data_preprocess, width=8).grid(
            row=2, columnspan=2
        )

    def _data_preprocess(self):
        # Check if input is empty
        if not self.field_var['l_freq'].get() and not self.field_var['h_freq'].get():
            raise ValidateException(window=self, message="No Input")

        l_freq = None
        h_freq = None
        if self.field_var['l_freq'].get().strip():
            l_freq = float(self.field_var['l_freq'].get())
        if self.field_var['h_freq'].get().strip():
            h_freq = float(self.field_var['h_freq'].get())
        
        try:
            self.return_data = self.preprocessor.data_preprocess(l_freq, h_freq)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))
        self.script_history.add_cmd(f'l_freq={repr(l_freq)}')
        self.script_history.add_cmd(f'h_freq={repr(h_freq)}')
        
        self.script_history.add_cmd((
            'study.preprocess(preprocessor=preprocessor.Filtering, '
            'l_freq=l_freq, h_freq=h_freq)'
        ))
        self.ret_script_history = self.script_history
        
        self.destroy()

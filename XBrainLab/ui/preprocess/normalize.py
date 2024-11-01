import tkinter as tk
from enum import Enum
from tkinter import ttk

from XBrainLab import preprocessor as Preprocessor

from ..base import ValidateException
from .base import PreprocessBase


class NormType(Enum):
    zscore = "z score"
    minmax = 'minmax'

class Normalize(PreprocessBase):
    command_label = "Normalize"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.Normalize(preprocessed_data_list)
        super().__init__(parent, "Normalize", preprocessor)
        norm_frame = ttk.LabelFrame(self, text="Normalize method")
        self.norm_ctrl = tk.StringVar()
        self.norm_ctrl.set(NormType.zscore.value)
        self.norm_zscore = tk.Radiobutton(
            norm_frame, text="Z score",
            value=NormType.zscore.value, variable=self.norm_ctrl
        )
        self.norm_minmax = tk.Radiobutton(
            norm_frame, text="Min-max",
            value=NormType.minmax.value, variable=self.norm_ctrl
        )

        norm_frame.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=10)
        self.norm_zscore.grid(row=0, column=0, sticky="w")
        self.norm_minmax.grid(row=0, column=1, sticky="w")
        tk.Button(
            self, text="Confirm",
            command=lambda win=self: win._data_preprocess(), width=8
        ).grid(row=1, columnspan=2)
    def _data_preprocess(self):
        try:
            norm_method = self.norm_ctrl.get()
            self.return_data = self.preprocessor.data_preprocess(norm_method)
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e
        self.script_history.add_cmd(f'norm_method="{norm_method}"')
        self.script_history.add_cmd(
            'study.preprocess(preprocessor=preprocessor.Normalize, norm=norm_method)'
        )
        self.ret_script_history = self.script_history

        self.destroy()

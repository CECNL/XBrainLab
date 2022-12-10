import tkinter as tk
import mne

from .base import PreprocessBase
from ..base import TopWindow, ValidateException, InitWindowValidateException

from XBrainLab import preprocessor as Preprocessor

class WindowEpoch(PreprocessBase):
    command_label = "Window Epoch"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.WindowEpoch(preprocessed_data_list)
        super().__init__(parent, "Window Epoch", preprocessor)
        
        self.rowconfigure([0, 1], weight=1)
        self.columnconfigure([1], weight=1)

        data_field = ["duration", "overlap", "baseline_tmin", "baseline_tmax"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Duration of each epoch (sec): ").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self, textvariable=self.field_var['duration']).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        tk.Label(self, text="Overlap between epochs (sec): ").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self, textvariable=self.field_var['overlap']).grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        tk.Button(self, text="Confirm", command=self._extract_epoch, width=8).grid(row=2, columnspan=2)

    def _extract_epoch(self):
        if not self.field_var['duration'].get().strip():
            raise ValidateException(window=self, message="No duration input")

        try:
            duration = float(self.field_var['duration'].get())
            overlap = 0.0 if self.field_var['overlap'].get() == "" else float(self.field_var['overlap'].get())
            self.return_data = self.preprocessor.data_preprocess(duration, overlap)
        except Exception as e:
            raise ValidateException(window=self, message=str(e))

        self.script_history.add_cmd(f'duration={repr(duration)}')
        self.script_history.add_cmd(f'overlap={repr(overlap)}')
        self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.WindowEpoch, duration=duration, overlap=overlap)')
        self.ret_script_history = self.script_history

        self.destroy()

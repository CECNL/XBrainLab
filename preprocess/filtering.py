from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
from copy import deepcopy
import tkinter as tk

class Filtering(TopWindow):
    command_label = "Filtering"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Filtering")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        self.return_data = None
        self.mne_data = deepcopy(self.preprocessed_data.mne_data)
        data_field = ["l_freq", "h_freq"]
        self.field_var = {key: tk.StringVar() for key in data_field}

        tk.Label(self, text="Lower pass-band edge: ").grid(row=2, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['l_freq'], bg="White").grid(row=2, column=1, sticky="w")
        tk.Label(self, text="Upper pass-band edge: ").grid(row=3, column=0, sticky="w")
        tk.Entry(self, textvariable=self.field_var['h_freq'], bg="White").grid(row=3, column=1, sticky="w")
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).grid(row=7, columnspan=2)

    def check_data(self):
        if not(type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
            raise InitWindowValidateException(window=self, message="No valid data is loaded")

    def _data_preprocess(self):
        # Check if input is empty
        if self.field_var['l_freq'].get() == "" and self.field_var['h_freq'].get() == "":
            raise InitWindowValidateException(window=self, message="No Input")

        for fn, mne_data in self.mne_data.items():
            l_freq = float(self.field_var['l_freq'].get()) if self.field_var['l_freq'].get() != "" else None
            h_freq = float(self.field_var['h_freq'].get()) if self.field_var['h_freq'].get() != "" else None
            self.mne_data[fn] = mne_data.filter(l_freq=l_freq, h_freq=h_freq)

        if type(self.preprocessed_data) == Raw:
            self.return_data = Raw(self.preprocessed_data.raw_attr, self.mne_data, self.preprocessed_data.raw_events, self.preprocessed_data.event_id)
        else:
            self.return_data = Epochs(self.preprocessed_data.epoch_attr, self.mne_data)
        self.destroy()

    def _get_result(self):
        return self.return_data

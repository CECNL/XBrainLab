from ..base import TopWindow, ValidateException, InitWindowValidateException
from ..dataset.data_holder import Raw, Epochs
from copy import deepcopy
import tkinter as tk

class Channel(TopWindow):
    command_label = "Channel"
    def __init__(self, parent, preprocessed_data):
        super().__init__(parent, "Select Channel")
        self.preprocessed_data = preprocessed_data
        self.check_data()

        self.return_data = None
        self.mne_data = deepcopy(self.preprocessed_data.mne_data)
        for k in self.mne_data.keys():
            ch_names = self.preprocessed_data.mne_data[k].ch_names
            break

        tk.Label(self, text="Choose Channels: ").pack()
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(self, selectmode="extended", yscrollcommand=scrollbar.set)
        for each_item in range(len(ch_names)):
            self.listbox.insert(tk.END, ch_names[each_item])
        self.listbox.pack(padx=10, pady=10, expand=True, fill="both")
        scrollbar.config(command = self.listbox.yview)
        tk.Button(self, text="Confirm", command=lambda win=self: self._data_preprocess(), width=8).pack()

    def check_data(self):
        if not(type(self.preprocessed_data) == Raw or type(self.preprocessed_data) == Epochs):
            raise InitWindowValidateException(window=self, message="No valid data is loaded")

    def _data_preprocess(self):
        self.select_channels = []
        for idx in list(self.listbox.curselection()):
            self.select_channels.append(self.listbox.get(idx))

        # Check if channel is selected
        if len(self.select_channels) == 0:
            raise InitWindowValidateException(window=self, message="No Channel is Selected")

        for fn, mne_data in self.mne_data.items():
            self.mne_data[fn] = mne_data.pick_channels(self.select_channels)

        if type(self.preprocessed_data) == Raw:
            self.return_data = Raw(self.preprocessed_data.raw_attr, self.mne_data, self.preprocessed_data.raw_events, self.preprocessed_data.event_id)
        else:
            self.return_data = Epochs(self.preprocessed_data.epoch_attr, self.mne_data)
        self.destroy()

    def _get_result(self):
        return self.return_data

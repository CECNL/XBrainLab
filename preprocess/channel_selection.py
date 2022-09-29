from .base import PreprocessBase
from ..base import ValidateException
import tkinter as tk

class ChannelSelection(PreprocessBase):
    command_label = "Channel Selection"
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Select Channel", preprocessed_data_list)
        mne_data = self.preprocessed_data_list[0].get_mne()
        ch_names = mne_data.ch_names

        tk.Label(self, text="Choose Channels: ").pack()
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(self, selectmode="extended", yscrollcommand=scrollbar.set)
        for ch in ch_names:
            self.listbox.insert(tk.END, ch)
        self.listbox.pack(padx=10, pady=10, expand=True, fill="both")
        scrollbar.config(command=self.listbox.yview)
        tk.Button(self, text="Confirm", command=self._data_preprocess, width=8).pack()

    def get_preprocess_desc(self, selected_channels):
        return f"Select {len(selected_channels)} Channel"
    
    def _data_preprocess(self):
        selected_channels = []
        for idx in list(self.listbox.curselection()):
            selected_channels.append(self.listbox.get(idx))

        # Check if channel is selected
        if len(selected_channels) == 0:
            raise ValidateException(window=self, message="No Channel is Selected")

        for preprocessed_data in self.preprocessed_data_list:
            preprocessed_data.get_mne().pick_channels(selected_channels)
            preprocessed_data.add_preprocess(self.get_preprocess_desc(selected_channels))

        self.return_data = self.preprocessed_data_list
        self.destroy()

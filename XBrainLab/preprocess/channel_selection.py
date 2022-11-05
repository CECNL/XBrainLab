from .base import PreprocessBase
from ..base import ValidateException
import tkinter as tk

class ChannelSelection(PreprocessBase):
    command_label = "Channel Selection"
    def __init__(self, parent, preprocessed_data_list):
        super().__init__(parent, "Select Channel", preprocessed_data_list)
        mne_data = self.preprocessed_data_list[0].get_mne()
        ch_names = mne_data.ch_names
        
        self.rowconfigure([1], weight=1)
        self.columnconfigure([0], weight=1)

        scrollbar = tk.Scrollbar(self)
        self.listbox = tk.Listbox(self, selectmode="extended", yscrollcommand=scrollbar.set)
        
        for ch in ch_names:
            self.listbox.insert(tk.END, ch)
        scrollbar.config(command=self.listbox.yview)
        
        tk.Label(self, text="Choose Channels: ").grid(row=0, column=0, columnspan=2)
        self.listbox.grid(row=1, column=0, padx=10, pady=10, sticky='news')
        scrollbar.grid(row=1, column=1, pady=10, sticky='news')
        tk.Button(self, text="Confirm", command=self._data_preprocess, width=8).grid(row=2, column=0, columnspan=2)

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

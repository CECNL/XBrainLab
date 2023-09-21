import tkinter as tk

from XBrainLab import preprocessor as Preprocessor

from ..base import ValidateException
from .base import PreprocessBase


class ChannelSelection(PreprocessBase):
    command_label = "Channel Selection"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.ChannelSelection(preprocessed_data_list)
        super().__init__(parent, "Select Channel", preprocessor)
        mne_data = preprocessor.get_preprocessed_data_list()[0].get_mne()
        ch_names = mne_data.ch_names

        self.rowconfigure([1], weight=1)
        self.columnconfigure([0], weight=1)

        scrollbar = tk.Scrollbar(self)
        self.listbox = tk.Listbox(
            self, selectmode="extended", yscrollcommand=scrollbar.set
        )

        for ch in ch_names:
            self.listbox.insert(tk.END, ch)
        scrollbar.config(command=self.listbox.yview)

        tk.Label(self, text="Choose Channels: ").grid(row=0, column=0, columnspan=2)
        self.listbox.grid(row=1, column=0, padx=10, pady=10, sticky='news')
        scrollbar.grid(row=1, column=1, pady=10, sticky='news')
        tk.Button(self, text="Confirm", command=self._data_preprocess, width=8).grid(
            row=2, column=0, columnspan=2
        )

    def _data_preprocess(self):
        selected_channels = [
            self.listbox.get(idx)
            for idx in list(self.listbox.curselection())
        ]

        try:
            self.return_data = self.preprocessor.data_preprocess(selected_channels)
        except Exception as e:
            raise ValidateException(window=self, message=str(e)) from e
        self.script_history.add_cmd(f'selected_channels={selected_channels!r}')
        self.script_history.add_cmd(
            'study.preprocess('
            'preprocessor=preprocessor.ChannelSelection, '
            'selected_channels=selected_channels)'
        )
        self.ret_script_history = self.script_history

        self.destroy()

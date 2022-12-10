import mne
import tkinter as tk
import tkinter.messagebox

from ..base import ValidateException
from .base import LoadBase, DataType


class LoadSet(LoadBase):
    command_label = "Import SET file (EEGLAB toolbox)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .set files")
        self.filetypes = [('eeg files (.set)', '*.set'),]

    def _load(self, filepath):
        data_type = None
        if self.type_ctrl.get() == DataType.RAW.value:
            try:
                selected_data = mne.io.read_raw_eeglab(filepath, uint16_codec='latin1', preload=True)                
                self.script_history.add_cmd(f"data = mne.io.read_raw_eeglab(filepath, uint16_codec='latin1', preload=True)")
                data_type = DataType.RAW.value
            except (TypeError):
                selected_data = mne.io.read_epochs_eeglab(filepath, uint16_codec='latin1')
                self.script_history.add_cmd(f"data = mne.io.read_epochs_eeglab(filepath, uint16_codec='latin1')")
                data_type = DataType.EPOCH.value
        else:
            try:
                selected_data = mne.io.read_epochs_eeglab(filepath, uint16_codec='latin1')
                self.script_history.add_cmd(f"data = mne.io.read_epochs_eeglab(filepath, uint16_codec='latin1')")
                data_type = DataType.EPOCH.value
            except (ValueError):
                selected_data = mne.io.read_raw_eeglab(filepath, uint16_codec='latin1', preload=True)
                self.script_history.add_cmd(f"data = mne.io.read_raw_eeglab(filepath, uint16_codec='latin1', preload=True)")
                data_type = DataType.RAW.value
        if data_type:
            self.check_data_type(data_type)
        return selected_data
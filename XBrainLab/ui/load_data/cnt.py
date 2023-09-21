import mne

from .base import LoadBase


class LoadCnt(LoadBase):
    command_label = "Import CNT file (Neuroscan)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .cnt files", lock_config_status=True)
        self.filetypes = [('eeg files (.cnt)', '*.cnt'),]

    def _load(self, filepath):
        data = mne.io.read_raw_cnt(filepath, preload=True)
        self.script_history.add_cmd(
            "data = mne.io.read_raw_cnt(filepath, preload=True)"
        )
        return data

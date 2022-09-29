import mne
from .base import LoadBase

class LoadCnt(LoadBase):
    command_label = "Import CNT file (Neuroscan)"

    def __init__(self, parent):
        super().__init__(parent, "Load data from .cnt files", lock_config_status=True)
        self.filetypes = (('.cnt files', '*.cnt'),)
        
    def _load(self, fn):
        return mne.io.read_raw_cnt(fn, preload=True)
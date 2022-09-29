import mne
from .base import LoadBase
from ..base import ValidateException
import tkinter as tk

class LoadEdf(LoadBase):
    command_label = "Import EDF/EDF+/GDF file (BIOSIG toolbox)"
    def __init__(self, parent, lock_config_status=True):
        super().__init__(parent, "Load data from .edf/.gdf files")
        self.filetypes = (('.edf files', '*.edf'),('.gdf files', '*.gdf'))
        
    def _load(self, fn):
        if '.edf' in fn:
            selected_data = mne.io.read_raw_edf(fn, preload=True)
        elif '.gdf' in fn:
            selected_data = mne.io.read_raw_gdf(fn, preload=True)
        else:
            raise ValidateException('Only EDF/GDF files are supported')
        return selected_data
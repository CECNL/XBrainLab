import mne

from .base import LoadBase
from ..base import ValidateException

class LoadEdf(LoadBase):
    command_label = "Import EDF/EDF+/GDF file (BIOSIG toolbox)"
    def __init__(self, parent):
        super().__init__(
            parent, 
            "Load data from .edf/.gdf files", lock_config_status=True
        )
        self.filetypes = [('eeg files (.edf, .gdf)', '*.edf *.gdf')]
        
    def _load(self, filepath):
        if '.edf' in filepath:
            selected_data = mne.io.read_raw_edf(filepath, preload=True)
            self.script_history.add_cmd(
                "data = mne.io.read_raw_edf(filepath, preload=True)"
            )
        elif '.gdf' in filepath:
            selected_data = mne.io.read_raw_gdf(filepath, preload=True)
            self.script_history.add_cmd(
                "data = mne.io.read_raw_gdf(filepath, preload=True)"
            )
        else:
            raise ValidateException(self, 'Only EDF/GDF files are supported')
        return selected_data
import scipy.io

from .base import LoadDict


class LoadMat(LoadDict):
    command_label = "Import MAT file (Matlab array)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .mat files")
        self.filetypes = [('eeg files(.mat)', '*.mat'),]
        self.script_history.add_import("import scipy.io")

    def _load(self, filepath):
        selected_data = scipy.io.loadmat(filepath)
        self.script_history.add_cmd("data = scipy.io.loadmat(filepath)")
        return self.handle_dict(filepath, selected_data)

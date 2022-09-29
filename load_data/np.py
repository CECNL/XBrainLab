import tkinter as tk
import numpy as np
from .base import LoadDict

class LoadNp(LoadDict):
    # npy: single array
    # npz: multiple arrays
    command_label = "Import NPY/NPZ file (Numpy array)"
    def __init__(self, parent):
        super().__init__(parent, "Load data from .npy/.npz files")
        self.filetypes = (('.npy files', '*.npy'), ('.npz files', '*.npz'))

    def _load(self, filepath):
        selected_data = np.load(filepath)
        if isinstance(selected_data, np.lib.npyio.NpzFile): # npz
            return self.handle_dict(filepath, selected_data)
        else: # npy
            return self.handle_array(filepath, selected_data)
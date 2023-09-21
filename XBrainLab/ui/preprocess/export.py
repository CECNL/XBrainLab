import tkinter as tk
from tkinter import filedialog

from XBrainLab import preprocessor as Preprocessor

from .base import PreprocessBase


class Export(PreprocessBase):
    command_label = "Export"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.Export(preprocessed_data_list)
        super().__init__(parent, "Export", preprocessor)
        self.withdraw()
        file_location = filedialog.askdirectory(title="Export Dataset")
        if file_location:
            self.return_data = self.preprocessor.data_preprocess(file_location)
            tk.messagebox.showinfo(parent=self, title='Finished', message='OK')
            self.script_history.add_cmd(f'filepath={file_location!r}')
            self.script_history.add_cmd(
                'study.preprocess(preprocessor=preprocessor.Export, '
                'filepath=filepath)'
            )
            self.ret_script_history = self.script_history
        self.destroy()

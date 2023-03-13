import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog

from ..base import ValidateException
from .base import PreprocessBase

from XBrainLab import preprocessor as Preprocessor

class Export(PreprocessBase):
    command_label = "Export"
    def __init__(self, parent, preprocessed_data_list):
        preprocessor = Preprocessor.Export(preprocessed_data_list)
        super().__init__(parent, "Export", preprocessor)
        self.withdraw()
        file_location = filedialog.askdirectory(title="Export Dataset" )
        if file_location:
            self.return_data = self.preprocessor.data_preprocess(file_location)
            tk.messagebox.showinfo(parent=self, title='Finished', message='OK')
            self.script_history.add_cmd(f'filepath={repr(file_location)}')
            self.script_history.add_cmd('study.preprocess(preprocessor=preprocessor.Export, filepath=filepath)')
            self.ret_script_history = self.script_history
        self.destroy()